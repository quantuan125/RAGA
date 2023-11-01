import streamlit as st
from utility.client import ClientDB
from utility.s3 import S3
from agent.miracle import MRKL
from streamlit_extras.colored_header import colored_header
from utility.authy import Login



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


class MainChat:
    @staticmethod
    def read_me_expander():
        st.markdown("""
        ## 🦜️ Welcome to Miracle! 
        Miracle is powered by **gpt-3.5-turbo**, specializing in construction, legal frameworks, and regulatory matters. 
        
        Below is a guide to help you navigate and understand the functionalities of this application better.
        """)
        
        colored_header(label="🛠️ Functionalities", color_name="blue-70", description="")
        
        st.markdown("""
        #### 1. **BR18 Feature** (Experimental)
        - **Enable BR18**: Integrate BR18 as part of Miracle's internal knowledge. You can toggle this feature in the sidebar.
        - **Search Types**:
            - **Header Search**: Searches by the headers in BR18. Recommend for specific queries
            - **Context Search**: Searches by content of paragraphs in BR18. Recommend for general queries

        #### 2. **Web Search Feature** (Experimental)
        - **Enable Web Search**: Integrate Google Search with up to 5 top results. You can adjust the number of results in the sidebar.

        #### 3. **Document Database**
        - **Upload & Process Document**: Upload PDFs as unstructured text and process them for Miracle to understand. Only one document can be processed at a time.
        - **Create Detailed Summary**: After processing a document, you can create a detailed summary of it. This might take 1-2 minutes.
        """)
        
        colored_header(label="📑 UI Interface", color_name="orange-70", description="")
        
        st.markdown("""
        #### 1. **Main Chat**: 
        - **View Source/Search Results**: Examine the results used by Miracle to produce its final answer.
        - **Clear Chat**: Resets the chat interface but does not reset functionalities.
                    
        #### 2. **PDF Display**: 
        - View your uploaded PDFs here. This tab only appears when a document is processed.
        """)
        
        colored_header(label="📜 SYSTEM PROMPT", color_name="yellow-70", description="")
        
        st.markdown("""
        For transparency, here is the initial prompt engineered for Miracle:

        ```
        You are Miracle, an expert in construction, legal frameworks, and regulatory matters.

        You have the following tools to answer user queries, but only use them if necessary. 

        Your primary objective is to provide responses that:
        1. Offer an overview of the topic, referencing the chapter and the section if relevant.
        2. List key points in bullet-points or numbered list format, referencing the clauses and their respective subclauses if relevant.
        3. Always match or exceed the details of the tool's output text in your answers. 
        4. Reflect back to the user's question and give a concise conclusion.
        5. If the search tool is used, you must always return the list of available URLs as part of your final answer. 

        Reminder: 
        Always try all your tools to find the answer to the user query

        Always self-reflect your answer based on the user's query and follows the list of response objective. 
        ```
        """)
        
        colored_header(label="🔗 Links", color_name="blue-green-70", description="")
        
        st.markdown("""
        - For any further assistance or more information, please contact <a href="mailto:qung@arkitema.com">qung@arkitema.com</a>.
        """, unsafe_allow_html=True)



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