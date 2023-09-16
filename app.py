from st_pages import Page, show_pages, add_page_title
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.mention import mention
import streamlit as st
from css import css


st.set_page_config(page_title="Home Page", page_icon="🏠", layout="wide")
st.title("Welcome to MRKL 🦜️")

st.markdown(css, unsafe_allow_html=True)
with st.empty():
    if 'openai_key' not in st.session_state:
        st.session_state.openai_key = ""

show_pages(
    [   
        Page("app.py", "Home", "🏠"),
        Page("configuration.py", "Configuration", "⚙️"),
        Page("mrkl.py", "Main Chat", "🦜️"),
    ]
)

with st.sidebar:
    with st.expander("Links", expanded=True):
        mention(
            label="Follow me on Twitter",
            icon="🐦",
            url="Your Twitter URL here"
        )
        mention(
            label="Check out my GitHub",
            icon="🐱",
            url="Your GitHub URL here"
        )
        mention(
            label="Connect on LinkedIn",
            icon="💼",
            url="Your LinkedIn URL here"
        )

st.header("About MRKL")
st.write("""
MRKL is your personal conversational agent, designed to assist you with a variety of tasks.
Whether you're looking to set reminders, get personalized recommendations, or just have a chat,
MRKL is here to make your life easier and more enjoyable.

**Features:**
- Real-time chat
- Information retrieval
- Task management
- And much more!

Created by [Your Name](Your LinkedIn or Portfolio link).

Follow me on [Twitter](Your Twitter link) | [GitHub](Your GitHub link)
""")

# Call to action
st.header("Get Started")
st.write("""
To begin interacting with MRKL, you'll need to enter your OpenAI API key for personalized, intelligent responses.
Don't worry, your API key is securely stored and only used to provide you with an enhanced conversational experience.
""")

# Redirect button to Configuration page
if st.button("Start the Application"):
    switch_page("Configuration")