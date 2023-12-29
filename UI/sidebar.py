import streamlit as st
from utility.client import ClientDB

class UI_Sidebar:

    @staticmethod
    def handle_collection_selection(existing_collections):

        def on_change_selected_collection():
            st.session_state.selected_collection_state = st.session_state.new_collection_state

            if st.session_state.new_collection_state and st.session_state.new_collection_state != "None":
                st.session_state.client_db = ClientDB(username=st.session_state.username, collection_name=st.session_state.selected_collection_state)
                # st.session_state.agent = MRKL()

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
