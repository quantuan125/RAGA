import streamlit as st
import os 
from RAG.ingestion_pipeline import Ingestion_Pipeline
from UI.explain import Explain_DL, Explain_DS, Explain_DP, Explain_DI, Explain_DE, Explain_VD
from utility.display import Ingestion_Stepper
from utility.config import IngestionConfigSettings, JSONConfigHandler
from streamlit_extras.row import row
import extra_streamlit_components as stx

class Ingestion_Settings:

    def update_last_selection(selection_type, selectbox_key):
        st.session_state.setting_last_selection = selection_type
        st.session_state.expanded_setting = selection_type + '_examples'

        temp_key = f'temp_{selectbox_key}'
        st.session_state[temp_key] = st.session_state[selectbox_key]
    
    def toggle_setting_expander(method_selection_key, expander_name):
        st.session_state.setting_last_selection = method_selection_key 
        st.session_state.expanded_setting = expander_name 

    def instances_dict_methods():
        Ingestion_Pipeline.initialize_ingestion_instances_and_mappings()

        document_loading_methods = {
            'None': None,
            'LangChain': Ingestion_Pipeline.document_loading_instance.document_loader_langchain,
            # 'Unstructured': Ingestion_Pipeline.document_loading_instance.document_loader_unstructured,
        }
        
        document_splitting_methods = {
            'None': None,
            'Recursive-Splitter': Ingestion_Pipeline.document_splitting_instance.recursive_splitter,
            'Character-Splitter': Ingestion_Pipeline.document_splitting_instance.character_splitter,
            'Markdown-Header-Splitter': Ingestion_Pipeline.document_splitting_instance.markdown_header_splitter
        }

        document_processing_methods = {
            'Clean-Chunks-Content': Ingestion_Pipeline.document_processing_instance.clean_chunks_content,
            'Customize-Document-Metadata': Ingestion_Pipeline.document_processing_instance.customize_document_metadata,
            'Filter-Short-Document': Ingestion_Pipeline.document_processing_instance.filter_short_documents,
            'Build-Index-Schema': Ingestion_Pipeline.document_processing_instance.build_index_schema,
            'Build-TOC': Ingestion_Pipeline.document_processing_instance.build_toc_from_documents,
        }

        document_indexing_methods = {
            'None': None, 
            'Summary-Indexing': Ingestion_Pipeline.document_indexing_instance.summary_indexing,
            'Parent-Document-Indexing': Ingestion_Pipeline.document_indexing_instance.parent_document_indexing,
        }
        
        document_embedding_methods = {
            'None': None,
            'OpenAI-Embedding': Ingestion_Pipeline.document_embedding_instance.openai_embedding_model
        }
        
        vector_database_methods = {
            'Chroma': Ingestion_Pipeline.vector_database_instance.build_chroma_vectorstore,
            'Redis': Ingestion_Pipeline.vector_database_instance.build_redis_vectorstore,
            'None': None  # Placeholder for other types
        }
        
        return document_loading_methods, document_splitting_methods, document_processing_methods, document_indexing_methods, document_embedding_methods, vector_database_methods

    def column_settings():

        st.subheader("Ingestion Settings")

        #uploaded_file = st.file_uploader("Upload your document (Markdown or PDF)", type=['md', 'pdf'])

        document_loading_methods, document_splitting_methods, document_processing_methods, document_indexing_methods, document_embedding_methods, vector_database_methods = Ingestion_Settings.instances_dict_methods()

        settings_row = row([0.8, 0.2], gap="small", vertical_align="bottom")

        default_document_loading_method = st.session_state.get('temp_selected_document_loading', 'None')

        settings_row.selectbox(
            "Choose a Document Loading method",
            options=list(document_loading_methods.keys()),
            index=list(document_loading_methods.keys()).index
            (default_document_loading_method),
            on_change=Ingestion_Settings.update_last_selection,
            args=('document_loading', 'selected_document_loading'),
            key='selected_document_loading'
        )

        if settings_row.button("⚙️", key='dl_setting_icon'):
            Ingestion_Settings.toggle_setting_expander('document_loading', 'document_loading_settings')

        st.session_state.document_loading_function = document_loading_methods[st.session_state.selected_document_loading]

        default_document_splitter = st.session_state.get('temp_selected_document_splitting', 'None')

        settings_row.selectbox(
            "Select the Document Splitting method:",
            options=list(document_splitting_methods.keys()),
            index=list(document_splitting_methods.keys()).index(default_document_splitter),
            on_change=Ingestion_Settings.update_last_selection,
            args=('document_splitting', 'selected_document_splitting'),
            key='selected_document_splitting'
        )

        if settings_row.button("⚙️", key='ds_setting_icon'):
            Ingestion_Settings.toggle_setting_expander('document_splitting', 'document_splitting_settings')

        st.session_state.document_splitting_function = document_splitting_methods[st.session_state.selected_document_splitting]


        default_document_processor = st.session_state.get('temp_selected_document_processing', [])

        settings_row.multiselect(
            "Select and order the Document Processing methods:",
            options=list(document_processing_methods.keys()),
            default=default_document_processor,
            on_change=Ingestion_Settings.update_last_selection,
            args=('document_processing', 'selected_document_processing'),
            key='selected_document_processing'
        )

        if settings_row.button("⚙️", key='dp_setting_icon'):
            Ingestion_Settings.toggle_setting_expander('document_processing', 'document_processing_settings')


        st.session_state.document_processing_function = [document_processing_methods[method] for method in st.session_state.selected_document_processing]

        
        default_document_indexer = st.session_state.get('temp_selected_document_indexing', 'None')

        settings_row.selectbox(
            "Select the Document Indexing methods:",
            options=list(document_indexing_methods.keys()),
            index=list(document_indexing_methods.keys()).index(default_document_indexer),
            on_change=Ingestion_Settings.update_last_selection,
            args=('document_indexing', 'selected_document_indexing'),
            key='selected_document_indexing'
        )

        if settings_row.button("⚙️", key='di_setting_icon'):
            Ingestion_Settings.toggle_setting_expander('document_indexing', 'document_indexing_settings')

        st.session_state.document_indexing_function = document_indexing_methods[st.session_state.selected_document_indexing]

        default_document_embedder = st.session_state.get('temp_selected_document_embedding', 'None')

        settings_row.selectbox(
            "Select the Embedding Model method:",
            options=list(document_embedding_methods.keys()),
            index=list(document_embedding_methods.keys()).index(default_document_embedder),
            on_change=Ingestion_Settings.update_last_selection,
            args=('document_embedding', 'selected_document_embedding'),
            key='selected_document_embedding'
        )

        if settings_row.button("⚙️", key='de_setting_icon'):
            Ingestion_Settings.toggle_setting_expander('document_embedding', 'document_embedding_settings')

        st.session_state.document_embedding_function = document_embedding_methods[st.session_state.selected_document_embedding]


        default_vector_database = st.session_state.get('temp_selected_vector_database', 'None')

        settings_row.selectbox(
            "Select the Vector Database method:",
            options=list(vector_database_methods.keys()),
            index=list(vector_database_methods.keys()).index(default_vector_database),
            on_change=Ingestion_Settings.update_last_selection,
            args=('vector_database', 'selected_vector_database'),
            key='selected_vector_database'
        )

        if settings_row.button("⚙️", key='vd_setting_icon'):
            Ingestion_Settings.toggle_setting_expander('vector_database', 'vector_database_settings')

        st.session_state.vector_database_function = vector_database_methods[st.session_state.selected_vector_database]


        with st.expander("Save Ingestion Configuration"):
            Ingestion_Settings.column_settings_save_config()

    def column_settings_save_config():
        config_folder_path = "json/config/ingestion"
        with st.form("save_ingestion_config"):
            st.session_state.ingestion_config_name = st.text_input("Enter configuration name", placeholder="config1")  

            if st.form_submit_button("Save Configuration"):
                # Create configuration dict based on user selections
                config_settings = IngestionConfigSettings.convert_fss_to_json_config(
                    st.session_state.document_loading_function,
                    st.session_state.document_splitting_function,
                    st.session_state.document_processing_function,
                    st.session_state.document_indexing_function,
                    st.session_state.document_embedding_function,
                    st.session_state.vector_database_function
                )
                # Save configuration to JSON
                config_file_name = st.session_state.ingestion_config_name if st.session_state.ingestion_config_name else "unknown_ingestion_config"
                config_file_path = os.path.join(config_folder_path, f"{config_file_name}.json")
                # Save configuration to JSON
                JSONConfigHandler.save_json_config(config_settings, config_file_path)  # The file path includes the directory and the file name
                st.success("Configuration Saved")


class Ingestion_Explain:
    @staticmethod
    def is_expanded(expander_name):
        auto_expansion = st.session_state.expanded_setting == expander_name
        return auto_expansion
    
    def set_display_flag():
        st.session_state.selected_column_option = 'Display'
    
    def settings_dict_methods():
        dl_explanation_methods = {
            'LangChain': (Explain_DL.document_loader_langchain, Explain_DL.document_loader_langchain_settings),
            # 'Unstructured': (Explain_DL.document_loader_unstructured, Explain_DL.document_loader_unstructured_settings),
        }
        
        ds_explanation_methods = {
            'Recursive-Splitter': (Explain_DS.recursive_splitter, Explain_DS.recursive_splitter_settings),
            'Character-Splitter': (Explain_DS.character_splitter, Explain_DS.character_splitter_settings),
            'Markdown-Header-Splitter': (Explain_DS.markdown_header_splitter, Explain_DS.markdown_header_splitter_settings),
        }

        dp_explanation_methods = {
            'Clean-Chunks-Content': (Explain_DP.clean_chunks_content, Explain_DP.clean_chunks_content_settings),
            'Customize-Document-Metadata': (Explain_DP.customize_document_metadata, Explain_DP.customize_document_metadata_settings),
            'Filter-Short-Document': (Explain_DP.filter_short_documents, Explain_DP.filter_short_documents_settings),
            'Build-Index-Schema': (Explain_DP.build_index_schema, Explain_DP.build_index_schema_settings),
            'Build-TOC': (Explain_DP.build_toc_from_documents, Explain_DP.build_toc_from_documents_settings),
        }

        di_explanation_methods = {
            'Summary-Indexing': (Explain_DI.summary_indexing, Explain_DI.summary_indexing_settings),
            'Parent-Document-Indexing': (Explain_DI.parent_document_indexing, Explain_DI.parent_document_indexing_settings),
        }
        
        de_explanation_methods = {
            'OpenAI-Embedding': (Explain_DE.openai_embedding_model, Explain_DE.openai_embedding_model_settings),
        }
        
        vd_explanation_methods = {
            'Chroma': (Explain_VD.build_chroma_vectorstore, Explain_VD.build_chroma_vectorstore_settings),
            'Redis': (Explain_VD.build_redis_vectorstore, Explain_VD.build_redis_vectorstore_settings),
        }

        return dl_explanation_methods, ds_explanation_methods, dp_explanation_methods, di_explanation_methods, de_explanation_methods, vd_explanation_methods


    def column_explain():

        dl_explanation_methods, ds_explanation_methods, dp_explanation_methods, di_explanation_methods, de_explanation_methods, vd_explanation_methods = Ingestion_Explain.settings_dict_methods()

        explain_header_row = row([0.13,0.1,0.9], gap="small", vertical_align="bottom")

        explain_header_row.subheader("Explain")
        explain_header_row.button("➡️", key="ingestion_go_to_display", on_click=Ingestion_Explain.set_display_flag)

        if st.session_state.setting_last_selection == 'document_loading' or st.session_state.expanded_setting == 'document_loading_settings':
            if st.session_state.selected_document_loading in dl_explanation_methods:
                example_method, settings_method = dl_explanation_methods[st.session_state.selected_document_loading]

                dl_expander_example = f"{st.session_state.selected_document_loading} Examples" if st.session_state.selected_document_loading != 'None' else "Document Loading Example"
                with st.expander(dl_expander_example, expanded=Ingestion_Explain.is_expanded('document_loading_examples')):
                    example_method()
                    
                dl_expander_setting = f"{st.session_state.selected_document_loading} Settings" if st.session_state.selected_document_loading != 'None' else "Document Loading Settings"
                with st.expander(dl_expander_setting, expanded=Ingestion_Explain.is_expanded('document_loading_settings')):
                    settings_method()

        if st.session_state.setting_last_selection == 'document_splitting' or st.session_state.expanded_setting == 'document_splitting_settings':
            if st.session_state.selected_document_splitting in ds_explanation_methods:
                example_method, settings_method = ds_explanation_methods[st.session_state.selected_document_splitting]

                ds_expander_example = f"{st.session_state.selected_document_splitting} Examples" if st.session_state.selected_document_splitting != 'None' else "Document Splitting Example"
                with st.expander(ds_expander_example, expanded=Ingestion_Explain.is_expanded('document_splitting_examples')):
                    example_method()

                ds_expander_setting = f"{st.session_state.selected_document_splitting} Settings" if st.session_state.selected_document_splitting != 'None' else "Document Splitting Settings"
                with st.expander(ds_expander_setting, expanded=Ingestion_Explain.is_expanded('document_splitting_settings')):
                    settings_method()

        # Example for document_processing:
        if st.session_state.setting_last_selection == 'document_processing' or st.session_state.expanded_setting == 'document_processing_settings':

                dp_expander_example = f"{st.session_state.selected_document_processing} Examples" if st.session_state.selected_document_processing != 'None' else "Document Processing Example"
                with st.expander(dp_expander_example, expanded=Ingestion_Explain.is_expanded('document_processing_examples')):
                    for method in st.session_state.selected_document_processing:
                        if method in dp_explanation_methods:
                            example_method, _ = dp_explanation_methods[method]
                            example_method()

                dp_expander_setting = f"{st.session_state.selected_document_processing} Settings" if st.session_state.selected_document_processing != 'None' else "Document Processing Settings"
                with st.expander(dp_expander_setting, expanded=Ingestion_Explain.is_expanded('document_processing_settings')):
                    for method in st.session_state.selected_document_processing:
                        if method in dp_explanation_methods:
                            _, settings_method = dp_explanation_methods[method]
                            settings_method()

        if st.session_state.setting_last_selection == 'document_indexing' or st.session_state.expanded_setting == 'document_indexing_settings':
            if st.session_state.selected_document_indexing in di_explanation_methods:
                example_method, settings_method = di_explanation_methods[st.session_state.selected_document_indexing]

                di_expander_example = f"{st.session_state.selected_document_indexing} Examples" if st.session_state.selected_document_indexing != 'None' else "Document Indexing Example"
                with st.expander(di_expander_example, expanded=Ingestion_Explain.is_expanded('document_indexing_examples')):
                    example_method()

                di_expander_setting = f"{st.session_state.selected_document_indexing} Settings" if st.session_state.selected_document_indexing != 'None' else "Document Indexing Settings"
                with st.expander(di_expander_setting, expanded=Ingestion_Explain.is_expanded('document_indexing_settings')):
                    settings_method()

        # ... Continue for document_embedding
        if st.session_state.setting_last_selection == 'document_embedding' or st.session_state.expanded_setting == 'document_embedding_settings':
            if st.session_state.selected_document_embedding in de_explanation_methods:
                example_method, settings_method = de_explanation_methods[st.session_state.selected_document_embedding]

                de_expander_example = f"{st.session_state.selected_document_embedding} Examples" if st.session_state.selected_document_embedding != 'None' else "Document Embedding Example"
                with st.expander(de_expander_example, expanded=Ingestion_Explain.is_expanded('document_embedding_examples')):
                    example_method()

                de_expander_setting = f"{st.session_state.selected_document_embedding} Settings" if st.session_state.selected_document_embedding != 'None' else "Document Embedding Settings"
                with st.expander(de_expander_setting, expanded=Ingestion_Explain.is_expanded('document_embedding_settings')):
                    settings_method()

        # ... Continue for vector_database
        if st.session_state.setting_last_selection == 'vector_database' or st.session_state.expanded_setting == 'vector_database_settings':
            if st.session_state.selected_vector_database in vd_explanation_methods:
                example_method, settings_method = vd_explanation_methods[st.session_state.selected_vector_database]

                vd_expander_example = f"{st.session_state.selected_vector_database} Examples" if st.session_state.selected_vector_database != 'None' else "Vector Database Example"
                with st.expander(vd_expander_example, expanded=Ingestion_Explain.is_expanded('vector_database_examples')):
                    example_method()

                vd_expander_setting = f"{st.session_state.selected_vector_database} Settings" if st.session_state.selected_vector_database != 'None' else "Vector Database Settings"
                with st.expander(vd_expander_setting, expanded=Ingestion_Explain.is_expanded('vector_database_settings')):
                    settings_method()


class Ingestion_Display:

    @staticmethod
    def set_settings_explain_flag():
        st.session_state.selected_column_option = 'Settings and Explain'

    @staticmethod
    def display_and_process_baseline_results(uploaded_file, test_generate_answer):
        baseline_container = st.container()
        with baseline_container.container():
            st.subheader("Baseline Ingestion")
            baseline_json_config = IngestionConfigSettings.get_baseline_json_config()

            baseline_fss_config = IngestionConfigSettings.mapping_ingestion_config(baseline_json_config)
            #st.write(baseline_fss_config)

            if test_generate_answer and uploaded_file:
                st.session_state.current_step = 0  # Update to the last step or appropriate step number
                baseline_ingestion_results = IngestionConfigSettings.run_ingestion_pipeline_with_config(uploaded_file, baseline_fss_config)

                st.session_state.baseline_ingestion_results = baseline_ingestion_results
                #st.write(baseline_ingestion_results)

            Ingestion_Stepper.display_containers(
                st.session_state.get('baseline_ingestion_results', {}),
                st.session_state.current_step,
                "Baseline Ingestion",
                key="baseline_ingestion_container",
                container_key="baseline_ingestion",
                config_info=baseline_json_config
            )
            st.divider()


    @staticmethod
    def display_and_process_customized_results(uploaded_file, test_generate_answer, config_folder_path):
        customized_container = st.container()
        with customized_container.container():
            for i in range(1, st.session_state.customized_containers + 1):
                container_row = row([0.8, 0.2], gap="small", vertical_align="bottom")
                container_row.subheader(f"Customized {i}")
                selected_config = container_row.selectbox(
                    "Select Configuration",
                    JSONConfigHandler.list_json_config_files(config_folder_path),
                    key=f"ingestion_config_select_{i}",
                )

                if selected_config != 'Default':
                    json_config = JSONConfigHandler.load_json_config(config_folder_path, selected_config)
                    fss_config = IngestionConfigSettings.mapping_ingestion_config(json_config)
                    #st.write(fss_config)
                else:
                    json_config = IngestionConfigSettings.convert_fss_to_json_config(
                        st.session_state.document_loading_function,
                        st.session_state.document_splitting_function,
                        st.session_state.document_processing_function,
                        st.session_state.document_indexing_function,
                        st.session_state.document_embedding_function,
                        st.session_state.vector_database_function
                    )
                    fss_config = IngestionConfigSettings.create_fss_config(
                        st.session_state.document_loading_function,
                        st.session_state.document_splitting_function,
                        st.session_state.document_processing_function,
                        st.session_state.document_indexing_function,
                        st.session_state.document_embedding_function,
                        st.session_state.vector_database_function
                    )
                    #st.write(fss_config)

                if test_generate_answer:
                    st.session_state.current_step = 0  # Assuming there's a step bar or similar UI component
                    customized_ingestion_results = IngestionConfigSettings.run_ingestion_pipeline_with_config(uploaded_file, fss_config)
                    st.session_state[f'customized_ingestion_results_{i}'] = customized_ingestion_results

                Ingestion_Stepper.display_containers(
                    st.session_state.get(f'customized_ingestion_results_{i}', {}),
                    st.session_state.current_step,
                    f"Customized {i}",
                    key=f"customized_container_{i}",
                    container_key=f"customized_{i}",
                    config_info=json_config
                )
                st.divider()

    def column_display():
        display_header_row = row([0.05, 0.05, 0.9], gap="small", vertical_align="bottom")
        display_header_row.button("⬅️", key="go_to_ingestion_settings_explain", on_click=Ingestion_Display.set_settings_explain_flag)
        display_header_row.subheader("Display", anchor=False)
        

        # display_row = row([0.2, 0.8, 0.2], gap="small", vertical_align="bottom")

        with st.container():
            uploaded_file = st.file_uploader("Upload Document", key="file_uploader")
            test_generate_answer = st.button("Start Ingestion")
            config_folder_path = "json/config"

            # Assuming similar steps for ingestion; adapt as needed
            pipeline_steps = ["Document Loading", "Document Splitting", "Document Processing", "Document Indexing", "Document Embedding", "Vector Databasing"]

            st.session_state.current_step = stx.stepper_bar(steps=pipeline_steps, lock_sequence=False)
            st.divider()

            if st.session_state.show_baseline:
                Ingestion_Display.display_and_process_baseline_results(uploaded_file, test_generate_answer)

            
            Ingestion_Display.display_and_process_customized_results(uploaded_file, test_generate_answer, config_folder_path)






