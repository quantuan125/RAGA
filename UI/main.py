import streamlit as st
from utility.client import ClientDB
from agent.miracle import MRKL



class Main:

    @staticmethod
    def handle_collection_selection(existing_collections):
        def on_change_selected_collection():
            st.session_state.selected_collection_state = st.session_state.new_collection_state
            st.session_state.s3_object_url = None 
            if st.session_state.new_collection_state is not None:
                actual_collection_name = f"{st.session_state.username}-{st.session_state.new_collection_state}"
                st.session_state.client_db = ClientDB(username=st.session_state.username, collection_name=actual_collection_name)
                st.session_state.agent = MRKL()

        display_collections = [col.split('-', 1)[1] for col in existing_collections if col is not None]

        # Insert a "None" option at the beginning of the display list
        display_collections.insert(0, "None")

        default_index = 0
        if st.session_state.get('selected_collection_state'):
            actual_collection_name = f"{st.session_state.username}-{st.session_state.selected_collection_state}"
            if actual_collection_name in existing_collections:
                default_index = existing_collections.index(actual_collection_name) + 1  # Adjusted for "None" at index 0

        selected_collection = st.selectbox(
            'Select a collection:',
            display_collections,
            index=default_index,
            key='new_collection_state',
            on_change=on_change_selected_collection
        )

        collection_object = None
        actual_collection_name = None
        if selected_collection and selected_collection != "None":
            actual_collection_name = f"{st.session_state.username}-{selected_collection}"
            collection_object = st.session_state.client_db.client.get_collection(actual_collection_name)

        return actual_collection_name, collection_object
    
    @staticmethod
    def get_display_collections(existing_collections):
        display_collections = [name.split('-', 1)[1] if '-' in name else name for name in existing_collections]
        return display_collections
    
    @staticmethod
    def create_new_collection():
        new_collection_name_input = st.text_input("Enter new collection name:")
        if st.button("Create Collection"):
            if new_collection_name_input:
                new_collection_name = f"{st.session_state.username}-{new_collection_name_input}"  # Prefix the username
                try:
                    st.session_state.client_db.client.create_collection(new_collection_name)
                    st.session_state.create_collection_message = f"Collection {new_collection_name_input} created successfully!"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error creating collection: {e}")

        if 'create_collection_message' in st.session_state:
            st.success(st.session_state.create_collection_message)
            # Clear the message from the session state after displaying it
            del st.session_state.create_collection_message

    @staticmethod
    def delete_collection(existing_collections):
        display_collections = Main.get_display_collections(existing_collections)
        delete_collection_selection = st.selectbox('Select a collection to delete:', display_collections, key='delete_collection')
        delete_collection = f"{st.session_state.username}-{delete_collection_selection}"
        if st.button("Delete Collection"):
            if delete_collection:
                try:
                    st.session_state.client_db.client.delete_collection(delete_collection)
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
        display_collections = Main.get_display_collections(existing_collections)
        rename_collection_selection = st.selectbox('Select a collection to rename:', display_collections, key='rename_collection')
        rename_collection = f"{st.session_state.username}-{rename_collection_selection}"
        new_name_input = st.text_input("Enter new name:")
        new_name = f"{st.session_state.username}-{new_name_input}"
        if st.button("Rename Collection"):
            if rename_collection and new_name_input:
                try:
                    collection = st.session_state.client_db.client.get_collection(rename_collection)
                    collection.modify(name=new_name)
                    st.session_state.rename_collection_message = f"Collection {rename_collection_selection} renamed to {new_name_input} successfully!"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error renaming collection: {e}")
        
        if 'rename_collection_message' in st.session_state:
            st.success(st.session_state.rename_collection_message)
            # Clear the message from the session state after displaying it
            del st.session_state.rename_collection_message
