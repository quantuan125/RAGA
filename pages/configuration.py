import streamlit as st
import os
from st_pages import add_page_title
from streamlit_extras.switch_page_button import switch_page
from UI.css import apply_css

def main():
    st.set_page_config(page_title="CONFIGURATION", page_icon="⚙️", layout="wide")
    apply_css()

    st.title("CONFIGURATION ⚙️")
    

    initial_api_key = st.session_state.openai_key if 'openai_key' in st.session_state else ""

    openai_api_key = st.text_input("Enter OpenAI API Key", value=initial_api_key, placeholder="Enter the OpenAI API key which begins with sk-", type="password")
    if openai_api_key:
        st.session_state.openai_key = openai_api_key
        #st.write(openai_api_key)
        st.success("API key has been entered. Press the Button to proceed to the main application")
        os.environ["OPENAI_API_KEY"] = openai_api_key

        if st.button("Proceed to Main Chat"):
            switch_page("Main Chat")
    else:
        st.info("Enter your OpenAI API key in the text box above and press 'Enter'. You can obtain the API key [here](https://platform.openai.com/account/api-keys)")


if __name__== '__main__':
    main()