import streamlit as st
from utility.client import ClientDB
from utility.authy import Login


class UI_Config:

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