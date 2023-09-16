from st_pages import Page, show_pages, add_page_title
import streamlit as st

st.set_page_config(page_title="Home Page", page_icon="🏠", layout="wide")
st.title("Home Page 🏠")

with st.empty():
    if 'openai_key' not in st.session_state:
        st.session_state.openai_key = None

show_pages(
    [   
        Page("app.py", "Home", "🏠"),
        Page("configuration.py", "Configuration", "⚙️"),
        Page("mrkl.py", "Main Chat", "🦜️"),
    ]
)