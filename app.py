from st_pages import Page, show_pages, add_page_title
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.mention import mention
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from UI.css import apply_css


st.set_page_config(page_title="Home Page", page_icon="🤖", layout="wide")
st.title("Welcome to RAGA 🤖")

apply_css()
with st.empty():
    if 'openai_key' not in st.session_state:
        st.session_state.openai_key = ""
    if 'username' not in st.session_state:
        st.session_state.username = None

show_pages(
    [   
        Page("app.py", "About", "🏠"),
        Page("pages/configuration.py", "Configuration", "⚙️"),
        Page("pages/dbm.py", "Database Management", "🗃️"),
        Page("pages/main.py", "Main", "🔍"),
        Page("pages/evaluation.py", "Evaluation", "📐"),
        Page("pages/playground.py", "Playground", "🕹️"),
    ]
)

with st.sidebar:
    with st.expander("Links", expanded=True):
        st.write("""
        **This app is made by Quan [QUNG] from COWI-ARKITEMA**
        """
        )
        mention(
            label="Check out my GitHub",
            icon="🐱",
            url="https://github.com/quantuan125"
        )
        mention(
            label="Connect on LinkedIn",
            icon="💼",
            url="https://www.linkedin.com/in/quan-nguyen-manh-anh-a00928170/"
        )

# Redirect button to Configuration page
if st.button("Proceed to Configuration"):
    switch_page("Configuration")