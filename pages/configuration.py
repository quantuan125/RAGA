import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from UI.css import apply_css
from utility.authy import Login
from utility.client import ClientDB
from utility.sessionstate import Init
from dotenv import load_dotenv
from UI.main import MainConfig



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
        #openai_api_key = st.text_input("Enter OpenAI API Key", value=initial_api_key, placeholder="Enter the OpenAI API key which begins with sk-", type="password")
        
        #if openai_api_key:
            #st.session_state.openai_key = openai_api_key
            #os.environ["OPENAI_API_KEY"] = openai_api_key


        if st.button("Sign In", key='signin_button'):
            Login.sign_in_process(username, password)

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
            selected_username, client_db_for_selected_user, sorted_collection_objects = MainConfig.setup_admin_environment()

            with st.expander("List of User Collections"):
                MainConfig.list_user_collections(sorted_collection_objects)

            with st.expander("Delete User Collection"):
                MainConfig.delete_user_collection(client_db_for_selected_user, sorted_collection_objects)
                
            with st.expander("Reset Client"):
                MainConfig.reset_client_for_user(selected_username, client_db_for_selected_user)
        else:
            st.warning("This section is restricted to the admin user only.")



    #st.write(st.session_state.username)
    #st.write(st.session_state.authentication)
    #st.write(st.session_state.client_db)

if __name__== '__main__':
    main()