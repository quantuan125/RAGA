import streamlit as st
import os 
from RAG.retrieval_pipeline import Retrieval_Pipeline
from UI.explain import Explain_QT, Explain_QC, Explain_PP, Explain_PR, Explain_VS
from utility.display import Retrieval_Stepper
from utility.config import RetrievalConfigSettings, JSONConfigHandler
from streamlit_extras.row import row
import extra_streamlit_components as stx


class Retrieval_Settings:

    def update_last_selection(selection_type, selectbox_key):
        st.session_state.setting_last_selection = selection_type
        st.session_state.expanded_setting = selection_type + '_examples'

        temp_key = f'temp_{selectbox_key}'
        st.session_state[temp_key] = st.session_state[selectbox_key]

        #print(f"Updating session state: {temp_key} = {st.session_state[selectbox_key]}")

        if selectbox_key == 'selected_query_construction':
            if st.session_state.selected_query_construction == 'Self-Query Construction':
                st.session_state.selected_vector_search = 'Self-Query Retrieval'
            else:
                st.session_state.selected_vector_search = 'None'
    
    def toggle_setting_expander(method_selection_key, expander_name):
        st.session_state.setting_last_selection = method_selection_key 
        st.session_state.expanded_setting = expander_name 
        #st.rerun()
    
    def instances_dict_methods():
        Retrieval_Pipeline.initialize_retrieval_instances_and_mappings()

        query_transformation_methods = {
            'None': None, 
            'Multi-Retrieval-Query': Retrieval_Pipeline.query_transformation_instance.multi_retrieval_query,
            'Rewrite-Retrieve-Read': Retrieval_Pipeline.query_transformation_instance.rewrite_retrieve_read,
            'Query-Extractor': Retrieval_Pipeline.query_transformation_instance.query_extractor,
            'Step-Back-Prompting': Retrieval_Pipeline.query_transformation_instance.step_back_prompting,
        }

        query_constructor_methods = {
            'None': None, 
            'Self-Query-Construction': Retrieval_Pipeline.query_construction_instance.self_query_constructor,
            'SQL-Query-Construction': Retrieval_Pipeline.query_construction_instance.sql_query_constructor,
            'SQL-Semantic-Query-Construction': Retrieval_Pipeline.query_construction_instance.sql_semantic_query_constructor
        }

        vector_search_methods = {
                'None': None, 
                'Base-Retrieval': Retrieval_Pipeline.vector_search_instance.base_retriever,
                'Reranking-Retrieval': Retrieval_Pipeline.vector_search_instance.reranking_retriever,
                'Self-Query-Retrieval': Retrieval_Pipeline.vector_search_instance.self_query_retriever,
                'Multi-Vector-Retrieval': Retrieval_Pipeline.vector_search_instance.multi_vector_retriever,
                'SQL-Retrieval': Retrieval_Pipeline.vector_search_instance.sql_retriever
            }
        
        post_processing_methods = {
                'Reranker': Retrieval_Pipeline.post_processing_instance.reranking,
                'Contextual-Compression': Retrieval_Pipeline.post_processing_instance.contextual_compression,
                'Filter-Top-Results': Retrieval_Pipeline.post_processing_instance.filter_top_documents
            }
        
        prompting_methods = {
            'None': None, 
            'Baseline-Prompt': Retrieval_Pipeline.prompting_instance.baseline_prompt, 
            'Custom-Prompt': Retrieval_Pipeline.prompting_instance.custom_prompt,
            'Step-Back-Prompt': Retrieval_Pipeline.prompting_instance.step_back_prompt,
            'SQL-Prompt': Retrieval_Pipeline.prompting_instance.sql_prompt
        }
        return query_transformation_methods, query_constructor_methods, vector_search_methods,post_processing_methods, prompting_methods

    #@st.cache_data(experimental_allow_widgets=True)
    def column_settings():
        st.subheader("Settings")

        query_transformation_methods, query_constructor_methods, vector_search_methods, post_processing_methods, prompting_methods = Retrieval_Settings.instances_dict_methods()

        settings_row = row([0.8, 0.2], gap="small", vertical_align="bottom")

        default_query_transformation_method = st.session_state.get('temp_selected_query_transformation', 'None')

        settings_row.selectbox(
            "Select the query transformation:",
            options=list(query_transformation_methods.keys()),
            index=list(query_transformation_methods.keys()).index(default_query_transformation_method),
            on_change=Retrieval_Settings.update_last_selection,
            args=('query_transformation', 'selected_query_transformation'),
            key='selected_query_transformation'
        )

        if settings_row.button("⚙️", key='qt_setting_icon'):
            Retrieval_Settings.toggle_setting_expander('query_transformation', 'query_transformation_settings')

        st.session_state.query_transformation_function = query_transformation_methods[st.session_state.selected_query_transformation]

        default_query_construction_method = st.session_state.get('temp_selected_query_construction', 'None')
        
        settings_row.selectbox(
            "Select the query constructor method:",
            options=list(query_constructor_methods.keys()),
            on_change=Retrieval_Settings.update_last_selection,
            index=list(query_constructor_methods.keys()).index(default_query_construction_method),
            args=('query_construction', 'selected_query_construction'),
            key='selected_query_construction'
        )

        if settings_row.button("⚙️", key='qc_setting_icon'):
            Retrieval_Settings.toggle_setting_expander('query_construction', 'query_construction_settings')

        st.session_state.query_construction_function = query_constructor_methods[st.session_state.selected_query_construction]

        default_vector_search_method = st.session_state.get('temp_selected_vector_search', 'None')

        settings_row.selectbox(
            "Select the vector search method:",
            options=list(vector_search_methods.keys()),
            index=list(vector_search_methods.keys()).index(default_vector_search_method),
            on_change=Retrieval_Settings.update_last_selection,
            args=('vector_search', 'selected_vector_search'),
            key='selected_vector_search'
        )

        if settings_row.button("⚙️", key='vs_setting_icon'):
            Retrieval_Settings.toggle_setting_expander('vector_search', 'vector_search_settings')

        st.session_state.vector_search_function = vector_search_methods[st.session_state.selected_vector_search]

        default_post_processing_methods = st.session_state.get('temp_selected_post_processing', [])

        settings_row.multiselect(
            "Select post processing methods:",
            options=list(post_processing_methods.keys()),
            default=default_post_processing_methods,
            on_change=Retrieval_Settings.update_last_selection,
            args=('post_processing', 'selected_post_processing'),
            key = 'selected_post_processing'
        )

        if settings_row.button("⚙️", key='pp_setting_icon'):
            Retrieval_Settings.toggle_setting_expander('post_processing', 'post_processing_settings')

        st.session_state.post_processing_function = [post_processing_methods[method] for method in st.session_state.selected_post_processing]

        default_prompting_method = st.session_state.get('temp_selected_prompting', 'None')

        settings_row.selectbox(
            "Select the prompting method:",
            options=list(prompting_methods.keys()),
            index=list(prompting_methods.keys()).index(default_prompting_method),
            on_change=Retrieval_Settings.update_last_selection,
            args=('prompting', 'selected_prompting'),
            key='selected_prompting'
        )

        if settings_row.button("⚙️", key='pr_setting_icon'):
            Retrieval_Settings.toggle_setting_expander('prompting', 'prompting_settings')

        st.session_state.prompting_function = prompting_methods[st.session_state.selected_prompting]
        
        # st.write(st.session_state.query_transformation_function)
        # st.write(st.session_state.query_construction_function)
        # st.write(st.session_state.vector_search_function)
        # st.write(st.session_state.post_processing_function)
        # st.write(st.session_state.prompting_function)

        with st.expander("Save Configuration"):
            config_folder_path = "json/config/retrieval"
            with st.form("save_config"):
                st.session_state.config_name = st.text_input("Enter configuration name", placeholder="config1")  

                if st.form_submit_button("Save Configuration"):
                    # Create configuration dict based on user selections
                    config_settings = RetrievalConfigSettings.convert_fss_to_json_config(
                        st.session_state.query_transformation_function,
                        st.session_state.query_construction_function,
                        st.session_state.vector_search_function,
                        st.session_state.post_processing_function,
                        st.session_state.prompting_function
                    )
                    
                    # Save configuration to JSON
                    config_file_name = st.session_state['config_name'] if st.session_state['config_name'] else "unknown_config"
                    config_file_path = os.path.join(config_folder_path, f"{config_file_name}.json")
                    # Save configuration to JSON
                    JSONConfigHandler.save_json_config(config_settings, config_file_path)  # The file path includes the directory and the file name
                    st.success("Configuration Saved")


class Retrieval_Explain:
    @staticmethod
    def is_expanded(expander_name):
        auto_expansion = st.session_state.expanded_setting == expander_name
        return auto_expansion

    def settings_dict_methods():
        qt_explanation_methods = {
            'Multi-Retrieval-Query': (Explain_QT.multi_retrieval_query, Explain_QT.multi_retrieval_query_settings),
            'Rewrite-Retrieve-Read': (Explain_QT.rewrite_retrieve_read, Explain_QT.rewrite_retrieve_read_settings),
            'Query-Extractor': (Explain_QT.query_extractor, Explain_QT.query_extractor_settings),
            'Step-Back-Prompting': (Explain_QT.step_back_prompting, Explain_QT.step_back_prompting_settings)
        }

        qc_explanation_methods = {
            'Self-Query-Construction': (Explain_QC.self_query_construction, Explain_QC.self_query_construction_settings),
        }

        vs_explanation_methods = {
            'Base-Retrieval': (Explain_VS.base_retriever, Explain_VS.base_retriever_settings),
            'Reranking-Retrieval': (Explain_VS.reranking_retriever, Explain_VS.reranking_retriever_settings),
            'Self-Query-Retrieval': (Explain_VS.self_query_retriever, Explain_VS.self_query_retriever_settings),
            'Multi-Vector-Retrieval': (Explain_VS.multi_vector_retriever, Explain_VS.multi_vector_retriever_settings),
        }

        pp_explanation_methods = {
            'Reranker': (Explain_PP.reranker, Explain_PP.reranker_settings),
            'Contextual-Compression': (Explain_PP.contextual_compression, Explain_PP.contextual_compression_settings),
            'Filter-Top-Results': (Explain_PP.filter_top_results, Explain_PP.filter_top_results_settings),
        }
        
        pr_explanation_methods = {
            'Baseline-Prompt': (Explain_PR.baseline_prompting, Explain_PR.baseline_prompting_settings),
            'Custom-Prompting': (Explain_PR.custom_prompting, Explain_PR.custom_prompting_settings),
        }

        return qt_explanation_methods, qc_explanation_methods, vs_explanation_methods, pp_explanation_methods, pr_explanation_methods

    def set_display_flag():
        st.session_state.selected_column_option = 'Display'

    def column_explain():

        qt_explanation_methods, qc_explanation_methods, vs_explanation_methods, pp_explanation_methods, pr_explanation_methods = Retrieval_Explain.settings_dict_methods()

        explain_header_row = row([0.13,0.1,0.9], gap="small", vertical_align="bottom")

        explain_header_row.subheader("Explain")
        explain_header_row.button("➡️", key="go_to_display", on_click=Retrieval_Explain.set_display_flag)

                
        if st.session_state.setting_last_selection == 'query_transformation' or st.session_state.expanded_setting == 'query_transformation_settings':

            if st.session_state.selected_query_transformation in qt_explanation_methods:
                example_method, settings_method = qt_explanation_methods[st.session_state.selected_query_transformation]

                qt_expander_example = f"{st.session_state.selected_query_transformation} Examples" if st.session_state.selected_query_transformation != 'None' else "Query Transformation Example"
                with st.expander(qt_expander_example, expanded=Retrieval_Explain.is_expanded('query_transformation_examples')):
                    example_method()
                    
                qt_expander_setting = f"{st.session_state.selected_query_transformation} Settings" if st.session_state.selected_query_transformation != 'None' else "Query Transformation Settings"
                with st.expander(qt_expander_setting, expanded=Retrieval_Explain.is_expanded('query_transformation_settings')):
                    settings_method()

        if st.session_state.setting_last_selection == 'query_construction' or st.session_state.expanded_setting == 'query_construction_settings':
            if st.session_state.selected_query_construction in qc_explanation_methods:
                example_method, settings_method = qc_explanation_methods[st.session_state.selected_query_construction]

                qc_expander_example = f"{st.session_state.selected_query_construction} Expamples" if st.session_state.selected_query_construction != 'None' else "Query Constructor Example"
                with st.expander(qc_expander_example, expanded=Retrieval_Explain.is_expanded('query_construction_examples')):
                    example_method()

                qc_expander_setting = f"{st.session_state.selected_query_construction} Settings" if st.session_state.selected_query_construction != 'None' else "Query Constructor Setting"
                with st.expander(qc_expander_setting, expanded=Retrieval_Explain.is_expanded('query_construction_settings')):
                    settings_method(st.session_state.selected_query_construction)
        
        if st.session_state.setting_last_selection == 'vector_search' or st.session_state.expanded_setting == 'vector_search_settings':
            if st.session_state.selected_vector_search in vs_explanation_methods:
                example_method, settings_method = vs_explanation_methods[st.session_state.selected_vector_search]
        
                vs_expander_example = f"{st.session_state.selected_vector_search} Examples" if st.session_state.selected_vector_search != 'None' else "Vector Search Example"
                with st.expander(vs_expander_example, expanded=Retrieval_Explain.is_expanded('vector_search_examples')):
                    example_method()
                
                vs_expander_setting = f"{st.session_state.selected_vector_search} Settings" if st.session_state.selected_vector_search != 'None' else "Vector Search Settings"
                with st.expander(vs_expander_setting, expanded=Retrieval_Explain.is_expanded('vector_search_settings')):
                    settings_method()

        if st.session_state.setting_last_selection == 'post_processing' or st.session_state.expanded_setting == 'post_processing_settings':

            pp_expander_example = f"{st.session_state.selected_post_processing} Expamples" if st.session_state.selected_post_processing != 'None' else "Post Processing Example"
            with st.expander(pp_expander_example, expanded=Retrieval_Explain.is_expanded('post_processing_examples')):
                for method in st.session_state.selected_post_processing:
                    if method in pp_explanation_methods:
                        example_method, _ = pp_explanation_methods[method]
                        example_method()

            pp_expander_setting = f"{st.session_state.selected_post_processing} Settings" if st.session_state.selected_post_processing != 'None' else "Post Processing Example"
            with st.expander(pp_expander_setting, expanded=Retrieval_Explain.is_expanded('post_processing_settings')):
                for method in st.session_state.selected_post_processing:
                    if method in pp_explanation_methods:
                        _, settings_method = pp_explanation_methods[method]
                        settings_method()

        if st.session_state.setting_last_selection == 'prompting' or st.session_state.expanded_setting == 'prompting_settings':
            if st.session_state.selected_prompting in pr_explanation_methods:
                example_method, settings_method = pr_explanation_methods[st.session_state.selected_prompting]

                pr_expander_example = f"{st.session_state.selected_prompting} Examples" if st.session_state.selected_prompting != 'None' else "Prompting Example"
                with st.expander(pr_expander_example, expanded=Retrieval_Explain.is_expanded('prompting_examples')):
                    example_method()
                
                pr_expander_setting = f"{st.session_state.selected_prompting} Settings" if st.session_state.selected_prompting != 'None' else "Prompting Settings"
                with st.expander(pr_expander_setting, expanded=Retrieval_Explain.is_expanded('prompting_settings')):
                    settings_method()


class Retrieval_Display:

    def set_settings_explain_flag():
        st.session_state.selected_column_option = 'Settings and Explain'

    @staticmethod
    def display_and_process_baseline_results(user_question, test_generate_answer):
        baseline_container = st.container()
        with baseline_container.container():
            st.subheader("Baseline")
            baseline_json_config = RetrievalConfigSettings.get_baseline_json_config()

            baseline_fss_config = RetrievalConfigSettings.mapping_retrieval_config(baseline_json_config)

            if test_generate_answer:
                st.session_state.current_step = 5
                baseline_pipeline_results = RetrievalConfigSettings.run_retrieval_pipeline_with_config(user_question, baseline_fss_config)

                st.session_state.baseline_pipeline_results = baseline_pipeline_results

            Retrieval_Stepper.display_containers(
                st.session_state.get('baseline_pipeline_results', {}),
                st.session_state.current_step,
                "Baseline",
                key="baseline_container",
                container_key="baseline",
                config_info=baseline_json_config
            )
            st.divider()

    @staticmethod
    def display_and_process_customized_results(user_question, test_generate_answer, config_folder_path):
        customized_container = st.container()
        with customized_container.container():
            for i in range(1, st.session_state.customized_containers + 1):
                container_row = row([0.8, 0.2], gap="small", vertical_align="bottom")
                container_row.subheader(f"Customized {i}")
                selected_config = container_row.selectbox(
                    "Select Configuration",
                    JSONConfigHandler.list_json_config_files(config_folder_path),
                    key=f"config_select_{i}",
                )

                if selected_config != 'Default':
                    json_config = JSONConfigHandler.load_json_config(config_folder_path, selected_config)
                    fss_config = RetrievalConfigSettings.mapping_retrieval_config(json_config)

                else:
                    json_config = RetrievalConfigSettings.convert_fss_to_json_config(
                    st.session_state.query_transformation_function,
                    st.session_state.query_construction_function,
                    st.session_state.vector_search_function,
                    st.session_state.post_processing_function,
                    st.session_state.prompting_function
                    )

                    fss_config = RetrievalConfigSettings.create_fss_config(
                    st.session_state.query_transformation_function,
                    st.session_state.query_construction_function,
                    st.session_state.vector_search_function,
                    st.session_state.post_processing_function,
                    st.session_state.prompting_function
                    )


                if test_generate_answer:
                    st.session_state.current_step = 5
                    customized_pipeline_results = RetrievalConfigSettings.run_retrieval_pipeline_with_config(user_question, fss_config)

                    st.session_state[f'customized_pipeline_results_{i}'] = customized_pipeline_results

                Retrieval_Stepper.display_containers(
                    st.session_state.get(f'customized_pipeline_results_{i}', {}),
                    st.session_state.current_step,
                    f"Customized {i}",
                    key=f"customized_container_{i}",
                    container_key=f"customized_{i}",
                    config_info=json_config
                )
                st.divider()
        

    def column_display():
        display_header_row = row([0.05,0.05,0.9], gap="small", vertical_align="bottom")
    
        display_header_row.button("⬅️", key="go_to_settings_explain", on_click=Retrieval_Display.set_settings_explain_flag)
        display_header_row.subheader("Display", anchor = False)

        display_row = row([0.2, 0.8, 0.2], gap="small", vertical_align="bottom")
        

        with st.container():

            display_row.selectbox("Preload Settings", ["None"], key="test")
            user_question = display_row.text_input("Type Something Here")
            #st.write(user_question)
            test_generate_answer = display_row.button("Generate")
            config_folder_path = "json/config"

            pipeline_steps = ["Query Transformation", "Query Construction", "Vector Search", "Post Processing", "Prompting", "Answer"]

            st.session_state.current_step = stx.stepper_bar(steps=pipeline_steps, lock_sequence=False)
            st.divider()

            if st.session_state.show_baseline:
                Retrieval_Display.display_and_process_baseline_results(user_question, test_generate_answer)
            
            Retrieval_Display.display_and_process_customized_results(user_question, test_generate_answer, config_folder_path)

            



