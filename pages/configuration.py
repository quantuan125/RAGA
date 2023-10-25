import streamlit as st
import os
from st_pages import add_page_title
from streamlit_extras.switch_page_button import switch_page
from UI.css import apply_css
from utility.authy import Login, s3htpasswd
from utility.client import ClientDB
from utility.sessionstate import Init
from UI.main import Main
from dotenv import load_dotenv



def main():
    load_dotenv()
    st.set_page_config(page_title="CONFIGURATION", page_icon="⚙️", layout="wide")
    apply_css()

    st.title("CONFIGURATION ⚙️")

    with st.empty():
        Init.initialize_session_state()

    login_tab, advanced_tab = st.tabs(["Log-in", "Advanced Settings"])
    
    with login_tab:
        initial_api_key = st.session_state.openai_key if 'openai_key' in st.session_state else ""

        st.subheader("Sign In")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        openai_api_key = st.text_input("Enter OpenAI API Key", value=initial_api_key, placeholder="Enter the OpenAI API key which begins with sk-", type="password")
        
        if openai_api_key:
            st.session_state.openai_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key


        if st.button("Sign In", key='signin_button'):
            Login.shared_sign_in_process(username, password)

            if st.session_state.authentication is True:
                st.session_state.client_db = ClientDB(username=st.session_state.username, collection_name=None, load_vector_store=False)
                

        if st.session_state.authentication is True:
            if st.button("Proceed to Main Chat"):
                st.session_state.authentication = False
                switch_page("Main Chat")

        # Checkbox to toggle the sign-up section
        if st.checkbox("New User? Sign Up Here"):
            st.subheader("Sign Up")
            signup_username = st.text_input("Choose a Username", placeholder="Enter a username for sign up")
            signup_password = st.text_input("Choose a Password", placeholder="Enter a password for sign up", type="password")
            
            if st.button("Sign Up"):
                Login.sign_up_process(signup_username, signup_password)

    with advanced_tab:
        st.subheader("Admin Functions")


        if 'username' in st.session_state and st.session_state.username == "admin":
            sorted_collections_objects = st.session_state.client_db.get_all_sorted_collections()

            with st.expander("List of Collections"):
                st.write(sorted_collections_objects)
        
            with st.expander("Delete Collection"):
                # Assume ClientDB is initialized with username 'admin'

                if not sorted_collections_objects:
                    st.warning("No collections found.")
                else:
                    all_collections = [col.name for col in sorted_collections_objects]
                    delete_collection_selections = st.multiselect(
                        'Select collections to delete:',
                        all_collections,
                        key='delete_all_collections'
                    )

                    if st.button(f"Delete Collection",key = "delete_collection_button"):
                        for delete_collection_selection in delete_collection_selections:
                            try:
                                st.session_state.client_db.client.delete_collection(delete_collection_selection)
                                st.session_state.delete_collection_message = f"Collection/collections deleted successfully!"
                            except Exception as e:
                                st.error(f"Error deleting collection: {e}")
                        st.experimental_rerun()

                    if 'delete_collection_message' in st.session_state:
                        st.success(st.session_state.delete_collection_message)
                        # Clear the message from the session state after displaying it
                        del st.session_state.delete_collection_message

        
        # Display a selection box with all usernames from the htpasswd file
            htpasswd_content = s3htpasswd.read_htpasswd()
            lines = htpasswd_content.split('\n')
            usernames = [line.split(":")[0] for line in lines if line.strip() and not line.startswith("admin")]

            selected_username = st.selectbox("Select an user:", usernames)

            with st.expander("Delete User Collection"):
                user_sorted_collections_objects = st.session_state.client_db.get_user_sorted_collections(selected_username)

                if not user_sorted_collections_objects:
                    st.warning("No collections found for selected user.")
                else:
                    user_collections = [col.name for col in user_sorted_collections_objects]
                    delete_collection_selection = st.selectbox(
                        'Select a collection to delete:',
                        user_collections,
                        key='delete_user_collection'
                    )

                    if st.button(f"Delete Collection", key = "delete_user_collection_button"):
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
                                
            with st.expander(f"Delete User"):
                if st.button(f"Delete {selected_username}"):
                    st.session_state.delete_user = "confirm_delete"
                
                if st.session_state.get('delete_user', "") == "confirm_delete":
                    st.warning(f"Are you sure you want to delete the user {selected_username}? This action cannot be undone and will remove all data associated with this user.")
                    
                    if st.button(f"Yes, Delete {selected_username}"):
                        # Delete the user account
                        Login.delete_user_account(selected_username)

                        # Set the 'delete' state to "success"
                        st.session_state.delete_user = "success"
                        
                        # Rerun the script to refresh the state and UI
                        st.experimental_rerun()

                    if st.session_state.get('delete_user', "") == "success":
                        st.success(f"User {selected_username} has been deleted!")
                        # Clear the 'delete' state
                        st.session_state.delete_user = None
            
            with st.expander("Reset Client"):
                if st.button(f"Reset Client for {selected_username}"):
                    st.session_state.reset_user = "confirm_reset"
                    
                if st.session_state.get('reset_user', "") == "confirm_reset":
                    st.warning(f"Are you sure you want to reset the client for {selected_username}? This action cannot be undone and will delete all collections and documents for the user.")
                    
                    if st.button(f"Yes, Reset Client for {selected_username}"):
                        
                        client_db = ClientDB(username=selected_username, collection_name=None, load_vector_store=False)
                        client_db.reset_client()

                        # Set the 'reset' state to "success"
                        st.session_state.reset_user = "success"
                        
                        st.experimental_rerun()

                if st.session_state.get('reset_user', "") == "success":
                    st.success(f"Client for {selected_username} has been reset!")
                    # Clear the 'reset' state
                    st.session_state.reset_user = None

        else:
            st.warning("This section is restricted to the admin user only.")



    #st.write(st.session_state.username)
    #st.write(st.session_state.authentication)
    #st.write(st.session_state.client_db)

if __name__== '__main__':
    main()