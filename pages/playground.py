from UI.sidebar import Sidebar
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
from pipeline.pipeline import Retrieval_Pipeline
import streamlit as st
from dotenv import load_dotenv
from UI.css import apply_css
from utility.sessionstate import Init
from UI.ui_main import Main
import langchain
from langchain.chains import HypotheticalDocumentEmbedder
import os
import datetime
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.vectorstores.redis import Redis
import json
from langchain.embeddings import OpenAIEmbeddings
from streamlit_elements import elements, mui, html

langchain.debug=True


def main():
    load_dotenv()
    st.set_page_config(page_title="Playground", page_icon="🕹️", layout="wide")
    apply_css()
    st.title("PLAYGROUND 🕹️")
    
    with st.empty():
        Init.initialize_session_state()
        Init.initialize_agent_state()
        Init.initialize_clientdb_state()
        if 'query_transformer' not in st.session_state:
            st.session_state.query_transformer_instance = QueryTransformer()
        if 'query_constructor' not in st.session_state:
            st.session_state.query_constructor_instance = QueryConstructor()
        # if 'vector_search_instance' not in st.session_state and st.session_state.vector_store is not None:
        #     st.session_state.vector_search_instance = VectorSearch()
        if 'prompting' not in st.session_state:
            st.session_state.prompting = Prompting()
        if 'vector_search_selection' not in st.session_state:
            st.session_state.vector_search_selection = 'None'

    with st.sidebar:
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

    with st.expander("View RAG Pipeline Diagram"):
        st.image("image/RAGAv3.png", caption="Retrieval-Augmented Generation (RAG) Pipeline Diagram")
        

    ingestion_tab, retrieval_tab = st.tabs(["Ingestion", "Retrieval"])

    with ingestion_tab:
        st.header("Indexing Setup for RAG Pipeline")
    
        #DOCUMENT LOADER
        with st.container():

            document_loader_methods = {
                'LangChain': DocumentLoader.document_loader_langchain,
                'Unstructured': DocumentLoader.document_loader_unstructured,
            }
            selected_document_loader = st.selectbox(
                "Choose a document loader method",
                options=list(document_loader_methods.keys()),
                index=0  # Default to 'Unstructured'
            )

            uploaded_file = st.file_uploader("Upload your document (Markdown or PDF)", type=['md', 'pdf'])

        #DOCUMENT SPLITTER
        with st.container():

            document_splitting_methods = {
                'None': None,
                'Recursive Splitter': DocumentSplitter.recursive_splitter,
                'Character Splitter': DocumentSplitter.character_splitter,
                'Markdown Header Splitter': DocumentSplitter.markdown_header_splitter
            }

            default_selection = 'Recursive Splitter'

            selected_document_splitter = st.selectbox(
                "Select the Document Splitting method:",
                options=list(document_splitting_methods.keys()),
                index=list(document_splitting_methods.keys()).index(default_selection)
            )

            # Store the selected document splitter method in session state
            st.session_state.document_splitter_function = document_splitting_methods[selected_document_splitter]

            if None != selected_document_splitter:
                with st.expander("Splitter Setting"):
                    chunk_size = st.slider(
                        "Choose the size of text chunks",
                        min_value=100, max_value=10000, value=3000, step=100
                    )
                    st.session_state.chunk_size = chunk_size
                    
                    chunk_overlap = st.slider(
                        "Choose the overlap between text chunks",
                        min_value=0, max_value=500, value=200, step=10
                    )
                    st.session_state.chunk_overlap = chunk_overlap

                    separator_display_mapping = {
                        "\\n\\n (Double Newline)": "\n\n",
                        "\\n (Newline)": "\n",
                        "Space": " ",
                        "Empty String": ""
                    }

                    # At the start of your Streamlit app, initialize selected_separators
                    if 'selected_separators' not in st.session_state:
                        st.session_state.selected_separators = list(separator_display_mapping.values())

                    # Then, in your UI logic, update selected_separators based on user interaction
                    separator_options = list(separator_display_mapping.keys()) + ["ALL"]
                    default_selections = list(separator_display_mapping.keys())  # Default to all separators selected

                    selected_separators_display = st.multiselect(
                        "Choose the separators for splitting",
                        options=separator_options,
                        default=default_selections
                    )

                    # If "ALL" is selected, use all separators, otherwise only selected ones
                    if "ALL" in selected_separators_display:
                        st.session_state.selected_separators = list(separator_display_mapping.values())
                    else:
                        st.session_state.selected_separators = [
                            separator_display_mapping[disp] 
                            for disp in selected_separators_display 
                            if disp in separator_display_mapping
                        ]

            if 'Markdown Header Splitter' == selected_document_splitter:
                default_headers = [("##", "h1_main"), ("###", "h2_chapter"), ("####", "h3_subchapter")]

                if 'headers_to_split_on' not in st.session_state:
                    st.session_state.headers_to_split_on = default_headers.copy()

                with st.expander("Markdown Header Splitter Settings:"):

                    # Function to add a new header
                    def add_header():
                        next_header_index = len(st.session_state.headers_to_split_on) + 1
                        next_header_sign = "#" * next_header_index
                        next_header_name = f"Header {next_header_index}"
                        st.session_state.headers_to_split_on.append((next_header_sign, next_header_name))

                    # Function to remove the last header
                    def remove_header():
                        if st.session_state.headers_to_split_on:
                            st.session_state.headers_to_split_on.pop()
                

                    # Display current headers and allow editing
                    for index, (header_sign, header_name) in enumerate(st.session_state.headers_to_split_on):
                        with st.container():
                            cols = st.columns([1, 4])
                            with cols[0]:
                                new_header_sign = st.text_input(f"Header sign {index+1}", value=header_sign)
                            with cols[1]:
                                new_header_name = st.text_input(f"Header name {index+1}", value=header_name)
                            if new_header_sign != header_sign or new_header_name != header_name:
                                st.session_state.headers_to_split_on[index] = (new_header_sign, new_header_name)


                    # Buttons to add/remove headers
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("Add header", on_click=add_header)
                    with col2:
                        st.button("Remove header", on_click=remove_header)

                    # Show the headers as they will be used in the splitter
                    st.write("Headers to split on:")
                    #for header in st.session_state.headers_to_split_on:
                        #st.write(f"{header[0]}: {header[1]}")
                    st.write(st.session_state.headers_to_split_on)

        #DOCUMENT PROCESSOR 
        with st.container():

            document_processing_methods = {
                'Clean Chunks Content': DocumentProcessor.clean_chunks_content,
                'Customize Document Metadata': DocumentProcessor.customize_document_metadata,
                'Filter Short Document': DocumentProcessor.filter_short_documents,
                'Build Index Schema': DocumentProcessor.build_index_schema,
                'Build TOC': DocumentProcessor.build_toc_from_documents,
            }

            # Selectbox for choosing the text splitter type
            selected_document_processor = st.multiselect(
                "Select and order the Document Processing methods:",
                options=list(document_processing_methods.keys())
            )

            if 'Customize Document Metadata' in selected_document_processor:
                all_removable_metadata_keys = ['source', 'file_name', 'file_type']
                default_removed_metadata_keys = ['source']

                # User selects which metadata components to remove
                remove_metadata_keys_selection = st.multiselect(
                    "Select metadata components to remove:",
                    options=all_removable_metadata_keys,
                    default=default_removed_metadata_keys
                )
                st.session_state.remove_metadata_keys = remove_metadata_keys_selection

                # All possible metadata components to add
                all_addable_metadata_keys = ['unique_id']
                default_added_metadata_keys = ['unique_id']
                
                # User selects which metadata components to add
                add_metadata_keys_selection = st.multiselect(
                    "Select metadata components to add:",
                    options=all_addable_metadata_keys,
                    default=default_added_metadata_keys
                )
                st.session_state.add_metadata_keys = add_metadata_keys_selection

                if 'unique_id' in add_metadata_keys_selection:
                    unique_id_options = ['uuid', 'file_name + uuid']
                    selected_unique_id_type = st.radio(
                        "Select the type of unique identifier to add:",
                        options=unique_id_options,
                        index=unique_id_options.index('file_name + uuid'),  # Default to 'file_name + uuid'
                        horizontal=True
                    )
                    st.session_state.selected_unique_id_type = selected_unique_id_type
                
        #REPRESENTATION INDEXER: 
        with st.container():
            document_indexer_methods = {
                'None': None, 
                'Summary Indexing': DocumentIndexer.summary_indexing,
                'Parent Document Indexing': DocumentIndexer.parent_document_indexing,
            }

            # Selectbox for choosing the text splitter type
            selected_document_indexer = st.selectbox(
                "Select the Document Indexing methods:",
                options=list(document_indexer_methods.keys())
            )

            st.session_state.document_indexer_function = document_indexer_methods[selected_document_indexer]

            if selected_document_indexer != 'None':
                with st.expander("Indexer Settings:"):
                # Add text input for file naming
                    custom_file_name = st.text_input("Enter a name for the output JSONL file (leave blank for auto-naming):")
                    if not custom_file_name:
                        # Generate a unique file name using a timestamp
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        custom_file_name = f"original_documents_{timestamp}.jsonl"
                    st.session_state.original_document_file_name = custom_file_name

        #EMBEDDER:
        with st.container():
            document_embedder_models = {
                'None': None,
                'OpenAI Embedding': DocumentEmbedder.openai_embedding_model
            }

            selected_document_embedder = st.selectbox(
                "Select the embedding model",
                options=list(document_embedder_models.keys()),
                index=0  # Default to OpenAI Embedding
            )

            st.session_state.document_embedder = document_embedder_models[selected_document_embedder]

        #VECTORSTORE
        with st.container():
            vectorstore_methods = {
                'Chroma': VectorStore.build_chroma_vectorstore,
                'Redis': VectorStore.build_redis_vectorstore,
                'None': None  # Placeholder for other types
            }

            selected_vectorstore = st.selectbox(
                "Select the vectorstore type",
                options=list(vectorstore_methods.keys()),  # Dynamically list available options
                index=0  # Default selection can be Chroma or Redis depending on preference
            )

            if None != selected_vectorstore:
                with st.expander("VectorStore Settings:"):
                    custom_collection_name = st.text_input("Enter a custom collection name:")
            
                    if custom_collection_name:
                        st.session_state.collection_name = custom_collection_name
                        st.success(f"Custom collection name set to: {custom_collection_name}")



        #INGESTION PIPELINE
        if st.button("Ingest Document"):
            ingestion = Retrieval_Pipeline.run_ingestion_pipeline(
                uploaded_file, 
                document_loading=document_loader_methods[selected_document_loader],
                document_splitting=document_splitting_methods[selected_document_splitter],
                document_processing = [document_processing_methods[method] for method in selected_document_processor],
                document_indexing = document_indexer_methods[selected_document_indexer],
                document_embedding=document_embedder_models[selected_document_embedder],
                vector_databasing=vectorstore_methods[selected_vectorstore],
                )
            
            st.session_state.custom_ingestion = ingestion

    with retrieval_tab:
        #QUERY TRANSFORMATION
        with st.container():

            query_transformer = st.session_state.query_transformer_instance

            query_transformation_methods = {
                'None': None, 
                'Multi-Retrieval Query': query_transformer.multi_retrieval_query,
                'Rewrite-Retrieve-Read': query_transformer.rewrite_retrieve_read,
                'Query Extractor': query_transformer.query_extractor,
                'Step-Back Prompting': query_transformer.step_back_prompting,
            }

            selected_query_transformation = st.selectbox(
                "Select the query transformation method:",
                options=list(query_transformation_methods.keys())
            )

            qt_expander_title = f"{selected_query_transformation} Explanation" if selected_query_transformation != 'None' else "Query Transformation Example"
            with st.expander(qt_expander_title):
                if selected_query_transformation == 'Multi-Retrieval Query':
                    st.markdown(
                        '''
                        ### Explanation 
                        
                        This method generates multiple variations of the original question to enhance the retrieval process by covering different phrasings or aspects of the query.

                        ### Example

                        **Input**: Original Question
                        ```
                        "How to deposit a cheque issued to an associate in my business into my business account?"
                        ```

                        **Output**: Multi Generated Questions
                        ```
                        1. "What is the process for depositing a cheque made out to an associate into my business account?"
                        2. "Can you explain how to deposit a cheque issued to an associate into my business account?"
                        3. "What are the steps to deposit a cheque made out to an associate into my business account?"
                        ```
                        '''
                    )
                elif selected_query_transformation == 'Rewrite-Retrieve-Read':
                    st.markdown(
                        '''
                        ### Explanation 
                        The Rewrite-Retrieve-Read method improves the retrieval process by first prompting an LLM to generate a more precise and focused search query from a user's initial vague or poorly structured question. This enhanced query is then used to retrieve relevant documents, which are analyzed by the LLM to formulate a comprehensive response.

                        ### Example
                        **Input**: Original Question
                        ```
                        "I wanna know about the stuff that makes computers think?"
                        ```
                        
                        **Output**: Rewritten Question
                        ```
                        "Key principles and technologies in artificial intelligence and machine learning for computer cognition"
                        ```
                        '''
                    )

                elif selected_query_transformation == 'Query Extractor':
                    st.markdown(
                        '''
                        ### Explanation
                        
                        The Query Extractor refines the user's natural language query to improve the retrieval of relevant documents from a vector database.
                        
                        ### Example
                        
                        **Input**: Original Question
                        ```
                        "What are the main factors contributing to urban pollution?"
                        ```
                        
                        **Output**: Extracted Question
                        ```
                        "urban pollution main contributing factors"
                        ```
                        '''
                    )


            if selected_query_transformation == 'Multi-Retrieval Query':
                with st.expander("Multi-Retrieval Query Settings"):
                    query_count = st.slider("Select the number of queries to generate", 1, 10, 3, 1)
                    st.session_state.query_count = query_count

                    default_mrq_prompt = """
                    You are an AI language model assistant. Your task is to generate {query_count} different versions of the given user question to retrieve relevant documents from a vector database.
                    
                    Provide these alternative questions separated by newlines.
                    Original question: {question}
                    """
                    mrq_prompt_template = st.text_area("Customize your Multi-Retrieval Query prompt:", value=default_mrq_prompt, height=200)
                    st.session_state.mrq_prompt_template = mrq_prompt_template

            #     def update_vector_search_original_query():
            #         st.session_state.vector_search_original_query = not st.session_state.vector_search_original_query

            #     st.checkbox(
            #     "Perform vector search on both original and step-back queries", 
            #     value=st.session_state.get('vector_search_original_query', True),
            #     on_change=update_vector_search_original_query
            # )

        #QUERY CONSTRUCTION     
        with st.container():

            query_constructor_instance = QueryConstructor()

            query_constructor_methods = {
                'None': None, 
                'Self-Query Construction': query_constructor_instance.self_query_constructor,
            }

            def on_query_constructor_change():
                if st.session_state.query_constructor_selection == 'Self-Query Construction':
                    st.session_state.vector_search_selection = 'Self-Query Retrieval'
                else:
                    st.session_state.vector_search_selection = 'None'

            selected_query_constructor = st.selectbox(
                "Select the query constructor method:",
                options=list(query_constructor_methods.keys()),
                on_change=on_query_constructor_change,
                key='query_constructor_selection'
            )

            qc_expander_title = f"{selected_query_constructor} Explanation" if selected_query_constructor != 'None' else "Query Constructor Example"
            with st.expander(qc_expander_title):
                if selected_query_constructor == 'Self-Query Construction':
                    st.markdown(
                        '''
                        **SelfQuery Retrieval:**
                        
                        This method translates natural language queries into structured queries using metadata filtering. This is particularly effective when dealing with vector databases that include structured data.
                        
                        **Example:**
                        
                        Given a natural language query like "What are movies about aliens in the year 1980", SelfQuery Retrieval will decompose the query into a semantic search term and logical conditions for metadata filtering, thus leveraging both semantic and structured data search capabilities.
                        '''
                    )
            
            with st.expander("Query Construction Settings:"):
                metadata_folder_path = 'json/metadata'
                schema_folder_path = 'json/schema'
                htso_folder_path = 'json/headers_info'
                toc_folder_path = 'json/toc'

                # List available metadata and schema files
                metadata_files = [f for f in os.listdir(metadata_folder_path) if f.endswith('.json')]
                schema_files = [f for f in os.listdir(schema_folder_path) if f.endswith('.json')]
                htso_files = [f for f in os.listdir(htso_folder_path) if f.endswith('.json')]
                toc_files = [f for f in os.listdir(toc_folder_path) if f.endswith('.json')]


                if selected_query_constructor == 'Self-Query Construction':
                    st.subheader("Configure Metadata Attribute and Description")
                    if 'document_content_description' not in st.session_state:
                        st.session_state.document_content_description = "Structured sections of the Danish Building Regulation 2018 document"
                    
                    st.session_state.document_content_description = st.text_area(
                    "Document Content Description",
                    value=st.session_state.document_content_description
                    )

                    # Selectbox for metadata field info files
                    selected_metadata_file = st.selectbox("Select Metadata Field Info File:", ['None'] + metadata_files)

                    if selected_metadata_file != 'None':
                        metadata_info_file_path = os.path.join(metadata_folder_path, selected_metadata_file)
                        # Load selected metadata field info
                        with open(metadata_info_file_path, 'r') as file:
                            st.session_state.metadata_field_info = json.load(file)
                            st.write(st.session_state.metadata_field_info)
                    else:
                        st.session_state.metadata_field_info = None

                    selected_toc_file = st.selectbox("Select Table of Contents File:", ['None'] + toc_files)

                    if selected_toc_file != 'None':
                        toc_file_path = os.path.join(toc_folder_path, selected_toc_file)
                        # Load selected ToC file
                        with open(toc_file_path, 'r') as file:
                            st.session_state.toc_content = json.load(file)
                            st.write(st.session_state.toc_content)
                    else:
                        st.session_state.toc_content = None

                    create_new_metadata_attr = st.checkbox("Create New Metadata Attribute", value=False)
                    if create_new_metadata_attr:
                        st.subheader("Create Metadata Attribute")

                        # Selectbox for index schema files
                        selected_schema_file = st.selectbox("Select Index Schema File:", ['None'] + schema_files)

                        selected_htso_file = st.selectbox(
                            "Select a 'Headers to Split On' JSON file:",
                            options=['None'] + htso_files
                        )

                        if st.button("Generate Metadata Field Info"):
                            if selected_schema_file != 'None' and selected_htso_file != 'None':
                                # Load index schema
                                json_schema_file_path = os.path.join(schema_folder_path, selected_schema_file)
                                with open(json_schema_file_path, 'r') as file:
                                    index_schema = json.load(file)

                                # Load headers to split on
                                json_htso_file_path = os.path.join(htso_folder_path, selected_htso_file)
                                with open(json_htso_file_path, 'r') as file:
                                    headers_info = json.load(file)

                                st.session_state.metadata_field_info = QueryConstructor.build_metadata_field_info(index_schema, headers_info)

                                # Save the generated metadata field info
                                timestamp_info = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                new_metadata_file_path = os.path.join(metadata_folder_path, f"metadata_field_info{timestamp_info}.json")

                                with open(new_metadata_file_path, 'w') as file:
                                    json.dump([attr_info.__dict__ for attr_info in st.session_state.metadata_field_info], file, indent=4)
                                
                                st.success(f"Metadata Field Info generated and saved as {new_metadata_file_path}")

                            # Function to add a new attribute
                        def add_attribute():
                            next_index = len(st.session_state.metadata_field_info) + 1
                            st.session_state.metadata_field_info.append(
                                AttributeInfo(name=f"Header{next_index}", description=f"Header {next_index} Content", type="string")
                            )

                            # Function to remove the last attribute
                        def remove_attribute():
                            if st.session_state.metadata_field_info:
                                st.session_state.metadata_field_info.pop()

                            # Display current attributes and allow editing
                        # for index, attribute_info in enumerate(st.session_state.metadata_field_info):
                        #     with st.container():
                        #         pass
                            # Buttons to add/remove attributes
                        col1, col2 = st.columns(2)
                        with col1:
                            st.button("Add attribute", on_click=add_attribute, key="add_attribute")
                        with col2:
                            st.button("Remove attribute", on_click=remove_attribute, key="remove_attribute")
                
                        # Optionally display the current configuration
                        # st.write("Current Metadata Fields:")
                        # for attribute_info in st.session_state.metadata_field_info:
                        #     st.write(f"Name: {attribute_info.name}, Description: {attribute_info.description}")

        #VECTOR SEARCH
        with st.container():

            vector_search_instance = VectorSearch()

            vector_search_methods = {
                'None': None, 
                'Base Retriever': vector_search_instance.base_retriever,
                'Reranking Retrieval': vector_search_instance.reranking_retriever,
                'Self-Query Retrieval': vector_search_instance.self_query_retriever,
                'Multi Vector Retrieval': vector_search_instance.multi_retriever_query
            }

            selected_vector_search = st.selectbox(
                "Select the vector search method:",
                options=list(vector_search_methods.keys()),
                index=list(vector_search_methods.keys()).index(st.session_state.vector_search_selection),
                key='vector_search_selection'
            )

            st.session_state.vector_search_function = vector_search_methods[selected_vector_search]

            vs_expander_title = f"{selected_vector_search} Explanation" if selected_vector_search != 'None' else "Vector Search Example"
            with st.expander(vs_expander_title):
                all_documents_str = '''
                    "doc1": "Climate change and economic impact.",
                    "doc2": "Public health concerns due to climate change.",
                    "doc3": "Climate change: A social perspective.",
                    "doc4": "Technological solutions to climate change.",
                    "doc5": "Policy changes needed to combat climate change.",
                    "doc6": "Climate change and its impact on biodiversity.",
                    "doc7": "Climate change: The science and models.",
                    "doc8": "Global warming: A subset of climate change.",
                    "doc9": "How climate change affects daily weather.",
                    "doc10": "The history of climate change activism."
                    '''

                if selected_vector_search == 'Base Retriever':
                    st.markdown(
                        '''
                        ### Explanation 

                        The Base Retriever method retrieves documents based on similarity to the input query using vector embeddings.

                        ### Example

                        **Documents**:
                        ```
                        "doc1": "Climate change and economic impact.",
                        "doc2": "Public health concerns due to climate change.",
                        "doc3": "Climate change: A social perspective.",
                        "doc4": "Technological solutions to climate change.",
                        "doc5": "Policy changes needed to combat climate change.",
                        "doc6": "Climate change and its impact on biodiversity.",
                        "doc7": "Climate change: The science and models.",
                        "doc8": "Global warming: A subset of climate change.",
                        "doc9": "How climate change affects daily weather.",
                        "doc10": "The history of climate change activism."
                        ```

                        **Input:** Original Question  
                        
                        ```
                        "impact of climate change"
                        ```

                        **Output:** Retrieved Documents (top k = 3)
                        
                        ```
                        "doc6": "Climate change and its impact on biodiversity."
                        "doc1": "Climate change and economic impact."
                        "doc9": "How climate change affects daily weather."
                        ```
                        '''
                    )
                elif selected_vector_search == 'Reranking Retriever':
                    st.markdown(
                        '''
                        ### Explanation 
                        This method first retrieves a set of documents similar to the query and then reranks them based on additional criteria to improve relevance.

                        ### Example

                        **Documents**:
                        
                        ```
                        "doc1": "Climate change and economic impact.",
                        "doc2": "Public health concerns due to climate change.",
                        "doc3": "Climate change: A social perspective.",
                        "doc4": "Technological solutions to climate change.",
                        "doc5": "Policy changes needed to combat climate change.",
                        "doc6": "Climate change and its impact on biodiversity.",
                        "doc7": "Climate change: The science and models.",
                        "doc8": "Global warming: A subset of climate change.",
                        "doc9": "How climate change affects daily weather.",
                        "doc10": "The history of climate change activism."
                        ```

                        **Input:** Original Question  
                        ```
                        impact of climate change
                        ```

                        **Output:** Top Ranked Documents  

                        ```
                        1. "doc2": "Public health concerns due to climate change.", score: 0.066
                        2. "doc3": "Climate change: A social perspective.", score: 0.064
                        3. "doc9": "How climate change affects daily weather.", score: 0.048
                        4. "doc6": "Climate change and its impact on biodiversity.", score: 0.033
                        5. "doc1": "Climate change and economic impact.", score: 0.017
                        6. "doc4": "Technological solutions to climate change.", score: 0.017
                        7. "doc5": "Policy changes needed to combat climate change.", score: 0.016
                        ```
                        '''
                    )
                
            if selected_vector_search != 'None':
                with st.expander("Retriever Settings"):
                    top_k_value = st.slider("Select the top k value for retrieval", 1, 20, 3, 1)
                    st.session_state.top_k_value = top_k_value

                    search_type = st.selectbox("Select the search type", ["similarity", "mmr", "similarity_score_threshold"])
                    st.session_state.search_type = search_type

                    if search_type == "mmr":
                        lambda_mult = st.slider("Select the lambda multiplier for MMR", 0.0, 1.0, 0.5, 0.01)
                        st.session_state.lambda_mult = lambda_mult

                        fetch_k = st.number_input("Select the fetch k value for MMR", 1, 50, 20, 1)
                        st.session_state.fetch_k = fetch_k

                    if search_type == "similarity_score_threshold":
                        score_threshold = st.slider("Set the similarity score threshold", 0.0, 1.0, 0.8, 0.01)
                        st.session_state.score_threshold = score_threshold

            if selected_vector_search == 'Reranking Retriever':
                with st.expander("Reranking Retriever Settings"):
                    # Slider to set the number of top reranked results
                    top_reranked_value = st.slider("Select the number of top reranked results", 1, 20, 5, 1)
                    st.session_state.top_reranked_value = top_reranked_value

                    # Slider to set the k value for reciprocal rank fusion
                    reciprocal_rank_k = st.slider("Select the k value for reciprocal rank fusion", 1, 100, 60, 1)
                    st.session_state.reciprocal_rank_k = reciprocal_rank_k

            if selected_vector_search == 'Multi Vector Retrieval':
                with st.expander("Multi Vector Retrieval Settings"):
                    inmemorystore_folder = os.path.join(os.getcwd(), "inmemorystore/indexed_documents")
                    jsonl_files = [f for f in os.listdir(inmemorystore_folder) if f.endswith('.jsonl')]
                    if jsonl_files:
                        options = ['None'] + jsonl_files
                        selected_file = st.selectbox("Select a JSONL file:", options)

                        if selected_file != 'None' and st.button("Load Selected File"):
                            file_path = os.path.join(inmemorystore_folder, selected_file)
                            st.session_state.inmemorystore = VectorSearch.load_documents_from_jsonl(file_path)
                            st.success(f"Loaded documents from {selected_file}")

                        # if 'inmemorystore' in st.session_state:
                        #     # Display information about the loaded documents
                        #     st.markdown("### Loaded Documents:")
                        #     st.write(st.session_state.inmemorystore.store)
                    else:
                        st.write("No JSONL files found in the inmemorystore folder.")
            
            #st.write(selected_vector_search)

        #POST-PROCESSING RETRIEVAL
        with st.container():

            post_processing_methods = {
                'Reranking': PostProcessor.prp_reranking,
                'Contextual Compression': PostProcessor.contextual_compression,
                'Filter Top Results': PostProcessor.filter_top_documents
            }

            selected_post_processor = st.multiselect(
                "Select post processing methods:",
                options=list(post_processing_methods.keys())
            )

            st.session_state.post_processing_function = [post_processing_methods[method] for method in selected_post_processor]

            if 'Filter Top Results' in selected_post_processor:
                st.session_state.filter_number = st.slider("Select the number of top results to filter", min_value=1, max_value=20, value=5, step=1)
            else:
                st.session_state.filter_number = 5 

            prp_expander_title = f"{selected_post_processor} Explanation" if selected_post_processor != 'None' else "Post-Retrieval Processing Example"
            with st.expander(prp_expander_title):
                if 'Reranking' in selected_post_processor:
                    st.markdown(
                        '''
                        ### Reranking

                        **Explanation:**  
                        Reranking involves reordering the initially retrieved documents based on additional relevance criteria or scores.

                        **Example:**  
                        Given initial retrieval results, reranking might prioritize documents more closely related to specific aspects of the query, resulting in a more refined set of top documents.
                        '''
                    )

                if 'Contextual Compression' in selected_post_processor:
                    st.markdown(
                        '''
                        ### Contextual Compression

                        **Explanation:**  
                        This method compresses the retrieved documents by removing redundant information and focusing on the most relevant content related to the query.

                        **Example:**  
                        From lengthy documents, only the sections or sentences that are most relevant to the query are retained, providing a concise and focused set of documents.
                        '''
                    )

                if 'Filter Top Results' in selected_post_processor:
                    st.markdown(
                        '''
                        ### Filter Top Results
                        **Explanation:**  
                        This process filters the retrieved or reranked documents to only include the top N results, based on a specified number.

                        **Example:**  
                        If the filter number is set to 5, only the top 5 documents (based on relevance or reranking scores) are selected for further processing.
                        '''
                    )

        #PROMPTING
        with st.container():

            prompting = st.session_state.prompting

            prompting_methods = {
                'None': None, 
                'Baseline Prompt': prompting.baseline_prompt, 
                'Custom Prompting': prompting.custom_prompt,
                'Step-Back Prompt': prompting.step_back_prompt
            }

            default_prompting_method = st.session_state.get('selected_prompting', 'None')

            selected_prompting = st.selectbox(
                "Select the prompting method:",
                options=list(prompting_methods.keys()),
                index=list(prompting_methods.keys()).index(default_prompting_method)
            )

            if selected_prompting == 'Custom Prompting':
                user_defined_prompt = st.text_area("Enter your custom prompt template here:")
                full_custom_prompt = user_defined_prompt + "\n\nContext: {context}\n\nQuestion: {question}"
                st.session_state.user_custom_prompt = user_defined_prompt
                st.session_state.full_custom_prompt = full_custom_prompt

            prompt_expander_title = f"{selected_prompting} Explanation" if selected_prompting != 'None' else "Prompting Example"
            with st.expander(prompt_expander_title):
                if selected_prompting == 'Baseline Prompt':
                    st.markdown(
                        '''
                        ### Baseline Prompt
                        **Explanation:**  
                        Uses a standard template to frame the context and question for generating the response.

                        **Example:**  
                        ```
                        Use the following pieces of context to answer the question at the end. 
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.

                        Context: {context}

                        Question: {question}
                        ```
                        '''
                    )
                elif selected_prompting == 'Custom Prompting':
                    custom_example = st.session_state.get('user_custom_prompt', 'No custom prompt defined.')
                    st.markdown(
                        f"""
                        ### Custom Prompting

                        **Explanation:**  
                        Allows users to define their own prompt template, giving more control over how the AI model generates responses.

                        **Example:**  
                        
                        ```
                        {custom_example}

                        Context: {{context}}
                        
                        Question: {{question}}
                        ```
                        
                        """
                    )

        user_question = st.text_input("Type Something Here")

        col1, col2 = st.columns(2)

        #RETRIEVAL PIPELINE
        generate_answer = st.button("Generate")
        if generate_answer:
            with col1:
                st.subheader("Baseline Pipeline")
                
                #baseline_answer = run_retrieval_pipeline(
                    #user_question,
                    #llm,
                    #retriever,
                    #query_transformation=None,
                    #vector_search=base_vector_search,
                    #reranking=False,
                    #compression=False,
                    #prompting=baseline_prompt
                #)

            with col2:
                st.subheader("Query Transformation Pipeline")
                st.markdown("### Query")
                st.write(user_question)

                custom_retrieval_answer = Retrieval_Pipeline.retrieval_pipeline(
                    user_question,
                    query_transformation=query_transformation_methods[selected_query_transformation],
                    query_construction = query_constructor_methods[selected_query_constructor],
                    vector_search=vector_search_methods[selected_vector_search],
                    post_processing=[post_processing_methods[method] for method in selected_post_processor],  # Passing the selected PRP methods
                    prompting=prompting_methods[selected_prompting]
                )

                st.session_state.custom_retrieval_answer = custom_retrieval_answer



    #st.write(st.session_state.top_k_value)
    #st.write(st.session_state.search_type)
    #st.write(st.session_state.score_threshold)
    #st.write(st.session_state.selected_vector_search)
    #st.write(st.session_state.client_db)
    st.write(st.session_state.vector_store)
    #st.write(selected_collection_name)
    #st.write(st.session_state.collection_name)
    #st.write(st.session_state.headers_to_split_on)
    #st.write(st.session_state.selected_document)
    #st.write(st.session_state.inmemorystore)
    

if __name__ == "__main__":
    main()