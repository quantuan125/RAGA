import streamlit as st
import os
from st_pages import add_page_title
from streamlit_extras.switch_page_button import switch_page

def main():
    st.set_page_config(page_title="CONFIGURATION", page_icon="⚙️", layout="wide")

    st.title("CONFIGURATION ⚙️")

    initial_api_key = st.session_state.openai_key if 'openai_key' in st.session_state else ""

    openai_api_key = st.text_input("Enter OpenAI API Key", value=initial_api_key, placeholder="Enter the OpenAI API key which begins with sk-", type="password")
    if openai_api_key:
        st.session_state.openai_key = openai_api_key
        st.success("API key has been entered. Press the Redirect Button to the main application")
        os.environ["OPENAI_API_KEY"] = openai_api_key

        if st.button("Redirect"):
            switch_page("Main Chat")


if __name__== '__main__':
    main()