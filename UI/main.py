import streamlit as st
from utility.client import ClientDB
from agent.miracle import MRKL
from streamlit_extras.colored_header import colored_header
from utility.authy import Login
import os 
from pipeline.pipeline import Retrieval_Pipeline, Ingestion_Pipeline
from UI.explain import Explain_QT, Explain_QC, Explain_PP, Explain_PR, Explain_VS
from UI.explain import Explain_DL, Explain_DS, Explain_DP, Explain_DI, Explain_DE, Explain_VD
from utility.display import Retrieval_Stepper, Ingestion_Stepper
from utility.config import RetrievalConfigSettings, IngestionConfigSettings, JSONConfigHandler
from streamlit_extras.row import row
import extra_streamlit_components as stx


class Main:

    @staticmethod
    def handle_collection_selection(existing_collections):

        def on_change_selected_collection():
            st.session_state.selected_collection_state = st.session_state.new_collection_state

            if st.session_state.new_collection_state and st.session_state.new_collection_state != "None":
                st.session_state.client_db = ClientDB(username=st.session_state.username, collection_name=st.session_state.selected_collection_state)
                st.session_state.agent = MRKL()

        display_collections = ["None"] + existing_collections

        default_index = 0
        if st.session_state.get('selected_collection_state') and st.session_state.selected_collection_state in display_collections:
            default_index = display_collections.index(st.session_state.selected_collection_state)


        selected_collection_name = st.selectbox(
            'Select a collection:',
            display_collections,
            index=default_index,
            key='new_collection_state',
            on_change=on_change_selected_collection
        )

        selected_collection_object = None
        if selected_collection_name and selected_collection_name != "None":
            selected_collection_object = st.session_state.client_db.client.get_collection(selected_collection_name)

        return selected_collection_name, selected_collection_object
    
    
    @staticmethod
    def create_new_collection():
        new_collection_name = st.text_input("Enter new collection name:")
        if st.button("Create Collection"):
            if new_collection_name:
                try:
                    st.session_state.client_db.client.create_collection(new_collection_name)
                    st.session_state.create_collection_message = f"Collection {new_collection_name} created successfully!"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error creating collection: {e}")
        
        if 'create_collection_message' in st.session_state:
            st.success(st.session_state.create_collection_message)
            # Clear the message from the session state after displaying it
            del st.session_state.create_collection_message

    @staticmethod
    def delete_collection(existing_collections):
        delete_collection_selection = st.selectbox('Select a collection to delete:', existing_collections, key='delete_collection')

        if st.button("Delete Collection"):
            if delete_collection_selection:
                try:
                    st.session_state.client_db.client.delete_collection(delete_collection_selection)
                    st.session_state.delete_collection_message = f"Collection {delete_collection_selection} deleted successfully!"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error deleting collection: {e}")

        if 'delete_collection_message' in st.session_state:
            st.success(st.session_state.delete_collection_message)
            # Clear the message from the session state after displaying it
            del st.session_state.delete_collection_message

    @staticmethod
    def rename_collection(existing_collections):   

        rename_collection_selection = st.selectbox('Select a collection to rename:', existing_collections, key='rename_collection')

        new_name = st.text_input("Enter new name:")

        if st.button("Rename Collection"):
            if rename_collection_selection and new_name:
                try:
                    collection = st.session_state.client_db.client.get_collection(rename_collection_selection)
                    collection.modify(name=new_name)
                    st.session_state.rename_collection_message = f"Collection {rename_collection_selection} renamed to {new_name} successfully!"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error renaming collection: {e}")
        
        if 'rename_collection_message' in st.session_state:
            st.success(st.session_state.rename_collection_message)
            # Clear the message from the session state after displaying it
            del st.session_state.rename_collection_message

class MainConfig:

    @staticmethod
    def setup_admin_environment():
        with open(Login.htpasswd_path, 'r') as file:
            htpasswd_content = file.read()

        lines = htpasswd_content.split('\n')
        usernames = [line.split(":")[0] for line in lines if line.strip() and not line.startswith("admin")]

        selected_username = st.selectbox("Select a user:", usernames)

        client_db_for_selected_user = ClientDB(username="admin", collection_name=None, admin_target_username=selected_username)

        sorted_collection_objects = client_db_for_selected_user.get_all_sorted_collections()
        
        return selected_username, client_db_for_selected_user, sorted_collection_objects

    @staticmethod
    def list_user_collections(sorted_collection_objects):
        st.write(sorted_collection_objects)

    @staticmethod
    def delete_user_collection(client_db_for_selected_user, sorted_collection_objects):
        if not sorted_collection_objects:
            st.warning("No collections found.")
            return

        all_collections = [col.name for col in sorted_collection_objects]
        delete_collection_selections = st.multiselect(
            'Select collections to delete:',
            all_collections,
            key='delete_all_collections'
        )

        if st.button(f"Delete Collection", key="delete_collection_button"):
            for delete_collection_selection in delete_collection_selections:
                try:
                    client_db_for_selected_user.client.delete_collection(delete_collection_selection)
                    st.session_state.delete_collection_message = f"Collection/collections deleted successfully!"
                except Exception as e:
                    st.error(f"Error deleting collection: {e}")
            st.experimental_rerun()

        if 'delete_collection_message' in st.session_state:
            st.success(st.session_state.delete_collection_message)
            del st.session_state.delete_collection_message

    @staticmethod
    def reset_client_for_user(selected_username, client_db_for_selected_user):
            if st.button(f"Reset Client for {selected_username}"):
                st.session_state.reset_user = "confirm_reset"
                
            if st.session_state.get('reset_user', "") == "confirm_reset":
                st.warning(f"Are you sure you want to reset the client for {selected_username}? This action cannot be undone and will delete all collections and documents for the user.")
                
                if st.button(f"Yes, Reset Client for {selected_username}"):
                    client_db_for_selected_user.reset_client()

                    # Set the 'reset' state to "success"
                    st.session_state.reset_user = "success"
                    
                    st.experimental_rerun()

            if st.session_state.get('reset_user', "") == "success":
                st.success(f"Client for {selected_username} has been reset!")
                # Clear the 'reset' state
                st.session_state.reset_user = None


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
        }

        vector_search_methods = {
                'None': None, 
                'Base-Retrieval': Retrieval_Pipeline.vector_search_instance.base_retriever,
                'Reranking-Retrieval': Retrieval_Pipeline.vector_search_instance.reranking_retriever,
                'Self-Query-Retrieval': Retrieval_Pipeline.vector_search_instance.self_query_retriever,
                'Multi-Vector-Retrieval': Retrieval_Pipeline.vector_search_instance.multi_vector_retriever
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
            'Step-Back-Prompt': Retrieval_Pipeline.prompting_instance.step_back_prompt
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
                    config_settings = IngestionConfigSettings.convert_fss_to_json_config(
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
            'Step-Back Prompting': (Explain_QT.step_back_prompting, Explain_QT.step_back_prompting_settings)
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

