import streamlit as st
from dotenv import load_dotenv
from UI.main import Main
from UI.css import apply_css
from UI.sidebar import Sidebar
from UI.explain import Explain_QT, Explain_QC, Explain_PP, Explain_PR, Explain_VS
from pipeline.rag.query_transformation import QueryTransformer
from pipeline.rag.query_construction import QueryConstructor
from pipeline.rag.vector_search import VectorSearch
from pipeline.rag.post_processing import PostProcessor
from pipeline.rag.prompting import Prompting
from pipeline.ingestion.document_loader import DocumentLoader
from pipeline.ingestion.document_splitter import DocumentSplitter
from pipeline.ingestion.document_processor import DocumentProcessor
from pipeline.ingestion.document_embedder import DocumentEmbedder
from pipeline.ingestion.vectorstore import VectorStore
from pipeline.ingestion.document_indexer import DocumentIndexer
from pipeline.pipeline import Pipeline
from utility.sessionstate import Init
from streamlit_extras.row import row
from streamlit_extras.grid import grid, example
from streamlit_extras.stylable_container import stylable_container
from langchain.vectorstores.redis import Redis
from langchain.embeddings import OpenAIEmbeddings
import extra_streamlit_components as stx
import os
import json
import langchain

langchain.debug=True

def display_pipeline_step(pipeline_results, current_step, step_title):
    if current_step == 0:
        st.write(f"{step_title} Transformed Question:", pipeline_results.get('query_transformation'))
    elif current_step == 1:
        st.write(f"{step_title} Constructed Query:", pipeline_results.get('query_construction'))
    elif current_step == 2:
        st.write(f"{step_title} Vector Search Results:", pipeline_results.get('vector_search'))
    elif current_step == 3:
        st.write(f"{step_title} Post Processing Results:", pipeline_results.get('post_processing'))
    elif current_step == 4:
        st.write(f"{step_title} Prompting:", pipeline_results.get('prompting'))
    elif current_step == 5:
        st.write(f"{step_title} Answer:", pipeline_results.get('answer'))

def update_last_selection(selection_type):
    st.session_state.setting_last_selection = selection_type
    st.session_state.expanded_setting = selection_type + '_examples'
    #st.rerun()

def toggle_setting_expander(method_selection_key, expander_name):
    st.session_state.setting_last_selection = method_selection_key 
    st.session_state.expanded_setting = expander_name 
    #st.rerun()
    

# Update the expander logic to check if it should be expanded
def is_expanded(expander_name):
    auto_expansion = st.session_state.expanded_setting == expander_name
    return auto_expansion

def display_pipeline_step_with_limited_height(pipeline_results, current_step, step_title, key):
    with stylable_container(
        key=key,
        css_styles=f"""
            {{
                max-height: 300px;  /* Set the maximum height you want */
                overflow-y: auto;   /* Add a scrollbar if content exceeds the container height */
                padding: 1rem;      /* Add some padding */
                border: 0px solid #ddd; /* Add a border for better visual separation */
            }}
        """,
    ):
        display_pipeline_step(pipeline_results, current_step, step_title)

    

def save_configuration(config_dict, file_path):
    # Convert dict to JSON string
    json_string = json.dumps(config_dict, indent=4)
    # Write to file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    with open(file_path, 'w') as f:
        f.write(json_string)

# Function to load a configuration
def list_config_files(config_folder_path):
    """List all JSON configuration files in the specified config directory."""
    files = [f for f in os.listdir(config_folder_path) if f.endswith('.json')]
    return ['Default'] + files  # Include 'Default' option

def load_configuration(config_folder_path, filename):
    """Load a JSON configuration file from the specified path and return the settings."""
    with open(os.path.join(config_folder_path, filename), 'r') as f:
        return json.load(f)
    


def main():
    load_dotenv()
    st.set_page_config(page_title="Main", page_icon="", layout="wide")
    apply_css()
    st.title("MAIN")

    with st.empty():
        #Init.initialize_session_state()
        Init.initialize_clientdb_state()
        if 'setting_last_selection' not in st.session_state:
            st.session_state.setting_last_selection = None
        if 'vector_search_selection' not in st.session_state:
            st.session_state.vector_search_selection = 'None'
        if 'expanded_setting' not in st.session_state:
            st.session_state.expanded_setting = None
        if 'baseline_pipeline_results' not in st.session_state:
            st.session_state.baseline_pipeline_results = {}
        if 'customized_pipeline_results' not in st.session_state:
            st.session_state.customized_pipeline_results = {}
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0



    with st.sidebar:
        #VECTORSTORE SELECTION
        with st.container():
            vectorstore_options = ['None', 'Chroma', 'Redis']

            selected_vectorstore = st.selectbox(
                "Select the vectorstore type",
                options=vectorstore_options,
                index=0,  # Default to None
            )

            if selected_vectorstore == 'None':
                st.session_state.vector_store = None
                st.warning("No vectorstore selected.")


            elif selected_vectorstore == 'Redis':
                redis_url = st.session_state.get('redis_url', 'redis://localhost:9000')
                redis_index_name = st.session_state.get('redis_index_name', 'user')


                # Load index schema
                index_schema_file_path = os.path.join("json/schema", "index_schema.json")
                with open(index_schema_file_path, 'r') as file:
                    redis_index_schema = json.load(file)

                st.session_state.redis_index_schema = redis_index_schema

                # Initialize Redis vectorstore
                st.session_state.vector_store = Redis(
                    redis_url=redis_url,
                    index_name=redis_index_name,
                    embedding=OpenAIEmbeddings(),
                    index_schema=redis_index_schema,
                )

            elif selected_vectorstore == 'Chroma':

                existing_collections = st.session_state.client_db.get_existing_collections()
                if not existing_collections:
                    st.warning("No collections available.")
                else:
                    selected_collection_name, selected_collection_object = Main.handle_collection_selection(existing_collections)
                    st.session_state.collection_name = selected_collection_name
            
                Sidebar.file_upload_and_ingest(st.session_state.client_db, selected_collection_name, selected_collection_object)
        

        with st.container():
            show_columns = st.sidebar.toggle("Show Settings and Explain Columns", value=True, key='show_columns')
            show_baseline = st.sidebar.toggle("Show Baseline Results", value=True, key='show_baseline')
            if 'customized_containers' not in st.session_state:
                st.session_state.customized_containers = 1  # Start with 1 customized container
            st.session_state.customized_containers = st.sidebar.number_input('Number of Customized Containers', min_value=1, max_value=4, value=st.session_state.customized_containers, step=1)

    with st.expander("View RAG Pipeline Diagram"):
        st.image("image/RAGAv3.png", caption="Retrieval-Augmented Generation (RAG) Pipeline Diagram")
        

    ingestion_tab, retrieval_tab = st.tabs(["Ingestion", "Retrieval"])

    with ingestion_tab:
        pass 

    with retrieval_tab:
        if show_columns:
            # Create columns for the three main sections
            col_settings, col_explain, col_display = st.columns([1.2, 2, 3])
            
            #QUERY TRANSFORMATION
            with st.container():
                with col_settings:

                    st.subheader("Settings")

                    settings_row = row([0.8, 0.2], gap="small", vertical_align="bottom")

                    query_transformatiion_instance = QueryTransformer()

                    query_transformation_methods = {
                        'None': None, 
                        'Multi-Retrieval Query': query_transformatiion_instance.multi_retrieval_query,
                        'Rewrite-Retrieve-Read': query_transformatiion_instance.rewrite_retrieve_read,
                        'Query Extractor': query_transformatiion_instance.query_extractor,
                        'Step-Back Prompting': query_transformatiion_instance.step_back_prompting,
                    }

                    selected_query_transformation = settings_row.selectbox(
                        "Select the query transformation:",
                        options=list(query_transformation_methods.keys()),
                        on_change=update_last_selection,
                        args=('query_transformation',)
                    )

                    if settings_row.button("⚙️", key='qt_setting_icon'):
                        toggle_setting_expander('query_transformation', 'query_transformation_settings')

                    st.session_state.query_transformation_function = query_transformation_methods[selected_query_transformation]
                
                with col_explain:

                    st.subheader("Explain")

                    qt_explanation_methods = {
                        'Multi-Retrieval Query': (Explain_QT.multi_retrieval_query, Explain_QT.multi_retrieval_query_settings),
                        'Rewrite-Retrieve-Read': (Explain_QT.rewrite_retrieve_read, Explain_QT.rewrite_retrieve_read_settings),
                        'Query Extractor': (Explain_QT.query_extractor, Explain_QT.query_extractor_settings),
                        'Step-Back Prompting': (Explain_QT.step_back_prompting, Explain_QT.step_back_prompting_settings)
                    }

                    if st.session_state.setting_last_selection == 'query_transformation' or st.session_state.expanded_setting == 'query_transformation_settings':

                        if selected_query_transformation in qt_explanation_methods:
                            example_method, settings_method = qt_explanation_methods[selected_query_transformation]

                            qt_expander_example = f"{selected_query_transformation} Examples" if selected_query_transformation != 'None' else "Query Transformation Example"
                            with st.expander(qt_expander_example, expanded=is_expanded('query_transformation_examples')):
                                example_method()
                                
                            qt_expander_setting = f"{selected_query_transformation} Settings" if selected_query_transformation != 'None' else "Query Transformation Settings"
                            with st.expander(qt_expander_setting, expanded=is_expanded('query_transformation_settings')):
                                settings_method()


            #QUERY CONSTRUCTION
            with st.container():   
                with col_settings:

                    query_cosntruction_instance = QueryConstructor()

                    query_constructor_methods = {
                        'None': None, 
                        'Self-Query Construction': query_cosntruction_instance.self_query_constructor,
                    }

                    def on_query_constructor_change(selection_type):
                        if st.session_state.query_constructor_selection == 'Self-Query Construction':
                            st.session_state.vector_search_selection = 'Self-Query Retrieval'
                        else:
                            st.session_state.vector_search_selection = 'None'
                        
                        st.session_state.setting_last_selection = 'query_construction'

                        st.session_state.expanded_setting = selection_type + '_examples'

                    selected_query_constructor = settings_row.selectbox(
                        "Select the query constructor method:",
                        options=list(query_constructor_methods.keys()),
                        on_change=on_query_constructor_change,
                        args=('query_construction',),
                        key='query_constructor_selection'
                    )

                    if settings_row.button("⚙️", key='qc_setting_icon'):
                        toggle_setting_expander('query_construction', 'query_construction_settings')

                    st.session_state.query_construction_function = query_constructor_methods[selected_query_constructor]

                with col_explain:

                    qc_explaination_methods = {
                        'Self-Query Construction': (Explain_QC.self_query_construction, Explain_QC.self_query_construction_settings),
                        # Add other methods as needed
                    }
                    
                    if st.session_state.setting_last_selection == 'query_construction' or st.session_state.expanded_setting == 'query_construction_settings':
                        if selected_query_constructor in qc_explaination_methods:
                            example_method, settings_method = qc_explaination_methods[selected_query_constructor]

                            qc_expander_example = f"{selected_query_constructor} Expamples" if selected_query_constructor != 'None' else "Query Constructor Example"
                            with st.expander(qc_expander_example, expanded=is_expanded('query_construction_examples')):
                                example_method()

                            qc_expander_setting = f"{selected_query_constructor} Settings" if selected_query_constructor != 'None' else "Query Constructor Setting"
                            with st.expander(qc_expander_setting, expanded=is_expanded('query_construction_settings')):
                                settings_method(selected_query_constructor)
            
            #VECTOR SEARCH
            with st.container():
                with col_settings:

                    vector_search_instance = VectorSearch()

                    vector_search_methods = {
                        'None': None, 
                        'Base Retrieval': vector_search_instance.base_retriever,
                        'Reranking Retrieval': vector_search_instance.reranking_retriever,
                        'Self-Query Retrieval': vector_search_instance.self_query_retriever,
                        'Multi Vector Retrieval': vector_search_instance.multi_retriever_query
                    }

                    selected_vector_search = settings_row.selectbox(
                        "Select the vector search method:",
                        options=list(vector_search_methods.keys()),
                        on_change=update_last_selection,
                        args=('vector_search',),
                        key='vector_search_selection'
                    )

                    if settings_row.button("⚙️", key='vs_setting_icon'):
                        toggle_setting_expander('vector_search', 'vector_search_settings')

                    st.session_state.vector_search_function = vector_search_methods[selected_vector_search]
                
                with col_explain:
                    
                    vs_explaination_methods = {
                        'Base Retrieval': (Explain_VS.base_retriever, Explain_VS.base_retriever_settings),
                        'Reranking Retrieval': (Explain_VS.reranking_retriever, Explain_VS.reranking_retriever_settings),
                        'Self-Query Retrieval': (Explain_VS.self_query_retriever, Explain_VS.self_query_retriever_settings),
                        'Multi Vector Retrieval': (Explain_VS.multi_vector_retriever, Explain_VS.multi_vector_retriever_settings),
                        # Add other methods as needed
                    }

                    if st.session_state.setting_last_selection == 'vector_search' or st.session_state.expanded_setting == 'vector_search_settings':
                        if selected_vector_search in vs_explaination_methods:
                                example_method, settings_method = vs_explaination_methods[selected_vector_search]
                        
                                vs_expander_example = f"{selected_vector_search} Examples" if selected_vector_search != 'None' else "Vector Search Example"
                                with st.expander(vs_expander_example, expanded=is_expanded('vector_search_examples')):
                                    example_method()
                                
                                vs_expander_setting = f"{selected_vector_search} Settings" if selected_vector_search != 'None' else "Vector Search Settings"
                                with st.expander(vs_expander_setting, expanded=is_expanded('vector_search_settings')):
                                    settings_method()


            #POST PROCESSING
            with st.container():
                with col_settings:

                    post_processing_methods = {
                        'Reranker': PostProcessor.prp_reranking,
                        'Contextual Compression': PostProcessor.contextual_compression,
                        'Filter Top Results': PostProcessor.filter_top_documents
                    }

                    selected_post_processor = settings_row.multiselect(
                        "Select post processing methods:",
                        options=list(post_processing_methods.keys()),
                        on_change=update_last_selection,
                        args=('post_processing',),
                    )

                    if settings_row.button("⚙️", key='pp_setting_icon'):
                        toggle_setting_expander('post_processing', 'post_processing_settings')

                    st.session_state.post_processing_function = [post_processing_methods[method] for method in selected_post_processor]
                    

                with col_explain:

                    pp_explaination_methods = {
                        'Reranker': (Explain_PP.reranker, Explain_PP.reranker_settings),
                        'Contextual Compression': (Explain_PP.contextual_compression, Explain_PP.contextual_compression_settings),
                        'Filter Top Results': (Explain_PP.filter_top_results, Explain_PP.filter_top_results_settings),
                    }

                    if st.session_state.setting_last_selection == 'post_processing' or st.session_state.expanded_setting == 'post_processing_settings':

                            pp_expander_example = f"{selected_post_processor} Expamples" if selected_post_processor != 'None' else "Post Processing Example"
                            with st.expander(pp_expander_example, expanded=is_expanded('post_processing_examples')):
                                for method in selected_post_processor:
                                    if method in pp_explaination_methods:
                                        example_method, _ = pp_explaination_methods[method]
                                        example_method()

                            pp_expander_setting = f"{selected_post_processor} Settings" if selected_post_processor != 'None' else "Post Processing Example"
                            with st.expander(pp_expander_setting, expanded=is_expanded('post_processing_settings')):
                                for method in selected_post_processor:
                                    if method in pp_explaination_methods:
                                        _, settings_method = pp_explaination_methods[method]
                                        settings_method()

            #PROMPTING
            with st.container():
                with col_settings:

                    prompting_instance = Prompting()

                    prompting_methods = {
                        'None': None, 
                        'Baseline Prompt': prompting_instance.baseline_prompt, 
                        'Custom Prompting': prompting_instance.custom_prompt,
                        'Step-Back Prompt': prompting_instance.step_back_prompt
                    }

                    default_prompting_method = st.session_state.get('selected_prompting', 'None')

                    selected_prompting = settings_row.selectbox(
                        "Select the prompting method:",
                        options=list(prompting_methods.keys()),
                        index=list(prompting_methods.keys()).index(default_prompting_method),
                        on_change=update_last_selection,
                        args=('prompting',),
                    )

                    if settings_row.button("⚙️", key='pr_setting_icon'):
                        toggle_setting_expander('prompting', 'prompting_settings')

                    st.session_state.prompting_function = prompting_methods[selected_prompting]

                with col_explain:

                    pr_explaination_methods = {
                        'Baseline Prompt': (Explain_PR.baseline_prompting, Explain_PR.baseline_prompting_settings),
                        'Custom Prompting': (Explain_PR.custom_prompting, Explain_PR.custom_prompting_settings),
                    }

                    if st.session_state.setting_last_selection == 'prompting' or st.session_state.expanded_setting == 'prompting_settings':
                        if selected_prompting in pr_explaination_methods:
                            example_method, settings_method = pr_explaination_methods[selected_prompting]

                            pr_expander_example = f"{selected_prompting} Examples" if selected_prompting != 'None' else "Prompting Example"
                            with st.expander(pr_expander_example, expanded=is_expanded('prompting_examples')):
                                example_method()
                            
                            pr_expander_setting = f"{selected_prompting} Settings" if selected_prompting != 'None' else "Prompting Settings"
                            with st.expander(pr_expander_setting, expanded=is_expanded('prompting_settings')):
                                settings_method()
            
            #BELOW
            with st.container():
                with col_settings:

                    with st.expander("Save Configuration"):

                        config_folder_path = "json/config"
                        st.session_state.config_name = st.text_input("Enter configuration name", placeholder="config1")  

                        def get_function_name(func):
                            """Return the name of the function or 'None' if the function is None."""
                            return func.__name__ if func is not None else 'None'

                        if st.button("Save Configuration"):
                            # Create configuration dict based on user selections
                            config_dict = {
                                'query_transformation': get_function_name(st.session_state.query_transformation_function),
                                'query_construction': get_function_name(st.session_state.query_construction_function),
                                'vector_search': get_function_name(st.session_state.vector_search_function),
                                'post_processing': [get_function_name(func) for func in st.session_state.post_processing_function],
                                'prompting': get_function_name(st.session_state.prompting_function),
                            }
                            # Save configuration to JSON
                            config_file_name = st.session_state['config_name'] if st.session_state['config_name'] else "unknown_config"
                            config_file_path = os.path.join(config_folder_path, f"{config_file_name}.json")
                            # Save configuration to JSON
                            save_configuration(config_dict, config_file_path)  # The file path includes the directory and the file name
                            st.success("Configuration Saved")

            with col_display:
                st.subheader("Display")
                display_row = row([0.8, 0.2], gap="small", vertical_align="bottom")

                with st.container():
                    user_question = display_row.text_input("Type Something Here")
                    generate_answer = display_row.button("Generate")

                    pipeline_steps = ["Query Transformation", "Query Construction", "Vector Search", "Post Processing", "Prompting", "Answer"]

                    if generate_answer:

                        def baseline_vector_search_function(questions):
                            # Ensure the questions argument is a list
                            if not isinstance(questions, list):
                                questions = [questions]
                            # Call the base retriever with the list of questions
                            return vector_search_instance.base_retriever(questions)
                        
                        #baseline_vector_search = vector_search_instance.base_retriever(user_question)
                        def baseline_prompting(combined_context, question):
                            return prompting_instance.baseline_prompt(combined_context, question)
        
                        baseline_pipeline_results = Pipeline.run_retrieval_pipeline(
                            user_question,
                            query_transformation=None,
                            query_construction=None,
                            vector_search=baseline_vector_search_function,
                            post_processing=[],
                            prompting=baseline_prompting,
                        )

                        st.session_state.baseline_pipeline_results = baseline_pipeline_results

                        customized_pipeline_results = Pipeline.run_retrieval_pipeline(
                            user_question,
                            query_transformation=query_transformation_methods[selected_query_transformation],
                            query_construction = query_constructor_methods[selected_query_constructor],
                            vector_search=vector_search_methods[selected_vector_search],
                            post_processing=[post_processing_methods[method] for method in selected_post_processor], 
                            prompting=prompting_methods[selected_prompting]
                        )

                        st.session_state.customized_pipeline_results = customized_pipeline_results

                    st.session_state.current_step = stx.stepper_bar(steps=pipeline_steps, lock_sequence=False)
            
                    # Display Baseline results
                    st.subheader("Baseline")
                    display_pipeline_step(st.session_state.get('baseline_pipeline_results', None), st.session_state.current_step, "Baseline")
                    
                    # Divider
                    st.divider()

                    # Display Customized results
                    st.subheader("Customized")
                    display_pipeline_step(st.session_state.get('customized_pipeline_results', None), st.session_state.current_step, "Customized")

        else: 
            col_display = st.columns([1])[0]
            with col_display:
                st.subheader("Display")
                display_row = row([0.2, 0.8, 0.2], gap="small", vertical_align="bottom")
                

                with st.container():
                    display_row.selectbox("Select Country", ["Germany", "Italy", "Japan", "USA"], key="test")
                    user_question = display_row.text_input("Type Something Here")
                    test_generate_answer = display_row.button("Generate")

                    pipeline_steps = ["Query Transformation", "Query Construction", "Vector Search", "Post Processing", "Prompting", "Answer"]

                    st.session_state.current_step = stx.stepper_bar(steps=pipeline_steps, lock_sequence=False)
                    st.divider()

                    container_row = row([0.2, 0.2, 0.8], gap="small", vertical_align="bottom")

                    if test_generate_answer:
                        
                        query_transformation_instance = QueryTransformer()
                        query_construction_instance = QueryConstructor()
                        vector_search_instance = VectorSearch()
                        post_processing_instance = PostProcessor()
                        prompting_instance = Prompting()


                        if show_baseline:
                            def baseline_vector_search_function(questions):
                                # Ensure the questions argument is a list
                                if not isinstance(questions, list):
                                    questions = [questions]
                                # Call the base retriever with the list of questions
                                return vector_search_instance.base_retriever(questions)
                            
                            #baseline_vector_search = vector_search_instance.base_retriever(user_question)
                            def baseline_prompting(combined_context, question):
                                return prompting_instance.baseline_prompt(combined_context, question)
            
                            baseline_pipeline_results = Pipeline.run_retrieval_pipeline(
                                user_question,
                                query_transformation=None,
                                query_construction=None,
                                vector_search=baseline_vector_search_function,
                                post_processing=[],
                                prompting=baseline_prompting,
                            )

                            st.session_state.baseline_pipeline_results = baseline_pipeline_results

                    # Display Baseline results
                    if show_baseline:
                        st.subheader("Baseline")
                        #container_row.selectbox("Select Country", ["Germany", "Italy", "Japan", "USA"], key="test_baseline")
                        with st.container():
                            display_pipeline_step_with_limited_height(
                                st.session_state.get('baseline_pipeline_results', {}), 
                                st.session_state.current_step, 
                                "Baseline",
                                key="baseline_container"
                            )
                        st.divider()

                    
                    query_transformation_instance = QueryTransformer()
                    query_construction_instance = QueryConstructor()
                    vector_search_instance = VectorSearch()
                    post_processing_instance = PostProcessor()
                    prompting_instance = Prompting()

                    pipeline_step_instances = {
                            'query_transformation': query_transformation_instance,
                            'query_construction': query_construction_instance,
                            'vector_search': vector_search_instance,
                            'post_processing': post_processing_instance,
                            'prompting': prompting_instance
                        }

                    pipeline_function_mappings = {
                        'query_transformation': {
                            'None': None,
                            'multi_retrieval_query': query_transformation_instance.multi_retrieval_query,
                            # Add other query transformation methods here
                        },
                        'query_construction': {
                            'None': None,
                            # If there are methods for query construction, map them here
                        },
                        'vector_search': {
                            'None': None,
                            'reranking_retriever': vector_search_instance.reranking_retriever,
                            # Add other vector search methods here
                        },
                        'post_processing': {
                            'None': None,
                            'contextual_compression': post_processing_instance.contextual_compression,
                            'filter_top_documents': post_processing_instance.filter_top_documents,
                            # Add other post processing methods here
                        },
                        'prompting': {
                            'None': None,
                            'baseline_prompt': prompting_instance.baseline_prompt,
                            # Add other prompting methods here
                        },
                    }

                    def get_pipeline_function(step_name, func_name):
                        """
                        Get the function based on the step name and function name.
                        The step_name is used to select the correct class instance,
                        and func_name is used to get the specific method.
                        """
                        # Retrieve the mapping for the specific pipeline step
                        function_mapping = pipeline_function_mappings.get(step_name)
                        if not function_mapping:
                            return None
                        
                        # Retrieve the function from the mapping based on the function name
                        return function_mapping.get(func_name)
                    
                    for i in range(1, st.session_state.customized_containers + 1):
                        container_row = row([0.2, 0.2, 0.8], gap="small", vertical_align="bottom")
                        container_row.subheader(f"Customized {i}")

                        config_folder_path = "json/config"

                        # This will display the configuration select box regardless of the button state
                        selected_config = container_row.selectbox(
                            "Select Configuration", 
                            list_config_files(config_folder_path), 
                            key=f"config_select_{i}"
                        )

                        selected_config = st.session_state[f'config_select_{i}']

                        # If a specific configuration is selected, load it and set the pipeline functions
                        if selected_config != 'Default':
                            # Load the selected configuration
                            config_settings = load_configuration(config_folder_path, selected_config)
                            st.write(config_settings)
                            
                            # Get pipeline functions based on the configuration settings
                            query_transformation_function = get_pipeline_function('query_transformation', config_settings['query_transformation'])
                            query_construction_function = get_pipeline_function('query_construction', config_settings['query_construction'])
                            vector_search_function = get_pipeline_function('vector_search', config_settings['vector_search'])
                            post_processing_functions = [get_pipeline_function('post_processing', func_name) for func_name in config_settings['post_processing'] if func_name != 'None']
                            prompting_function = get_pipeline_function('prompting', config_settings['prompting'])

                        else:
                            # Use default session state functions
                            query_transformation_function = st.session_state.query_transformation_function
                            query_construction_function = st.session_state.query_construction_function
                            vector_search_function = st.session_state.vector_search_function
                            post_processing_functions = st.session_state.post_processing_function
                            prompting_function = st.session_state.prompting_function

                        if test_generate_answer:

                            # Run the pipeline with the current configuration
                            customized_pipeline_results = Pipeline.run_retrieval_pipeline(
                                user_question,
                                query_transformation=query_transformation_function,
                                query_construction=query_construction_function,
                                vector_search=vector_search_function,
                                post_processing=post_processing_functions,
                                prompting=prompting_function
                            )

                            # Update the results in the session state
                            st.session_state[f'customized_pipeline_results_{i}'] = customized_pipeline_results

                        display_pipeline_step_with_limited_height(
                                st.session_state.get(f'customized_pipeline_results_{i}', {}), 
                                st.session_state.current_step, 
                                f"Customized {i}",
                                key=f"customized_container_{i}"
                            )

                        st.divider()

                    
                    # for i in range(1, st.session_state.customized_containers + 1):
                    #         display_pipeline_step_with_limited_height(
                    #             st.session_state.get(f'customized_pipeline_results_{i}', {}), 
                    #             st.session_state.current_step, 
                    #             f"Customized {i}",
                    #             key=f"customized_container_{i}"
                    #         )

                    #         st.divider()
    
                #example()


    #st.write(st.session_state.setting_last_selection)
    #st.write(st.session_state.expanded_setting)


if __name__ == "__main__":
    main()

