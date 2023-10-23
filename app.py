from st_pages import Page, show_pages, add_page_title
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.mention import mention
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from UI.css import apply_css


st.set_page_config(page_title="Home Page", page_icon="🏠", layout="wide")
st.title("Welcome to Miracle 🦜️")

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
        Page("pages/chat.py", "Main Chat", "🦜️"),
        Page("pages/dbm.py", "Database Management", "🗃️"),
        Page("pages/agent.py", "Agent Settings", "🤖"),
        Page("pages/br18.py", "BR18", "📚")
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

st.header("About Miracle")
st.write("""
         
Miracle is your intelligent conversational agent, purpose-built to redefine how we interact with information within complex sectors like construction, legal frameworks, and regulatory matters. Conceived and developed within the forward-thinking AI community at COWI-ARKITEMA, Miracle is a testament to our shared vision for a more advanced and interactive digital ecosystem.

         
### What Makes Miracle Unique?
         

Miracle is not just another chatbot. It's a highly specialized agent equipped with the capability to pull real-time, context-rich data, and engage in meaningful dialogues about that information. Miracle adapts a human-like approach to problem-solving within specific domains. It thinks, it reasons, and it refines its answers, offering:

- **Informed Dialogues**: Topic-specific conversations backed by up-to-date data.
- **Contextual Interactions**: Ability to understand and discuss documents uploaded to the system.
- **Specialized Knowledge**: Expertise in niche areas such as the Danish Building Regulation 2018 (BR18).

         
### Objective and Professionalism
         

In every interaction, Miracle maintains a high standard of professionalism while focusing on delivering:

1. Comprehensive overviews of topics, citing relevant chapters and sections.
2. Key points in a structured list format, with references to specific clauses and subclauses.
3. Detailed and well-reflected answers that outshine the raw outputs from its internal tools.

         
### Your Feedback Matters
         

Since Miracle is still evolving and not yet ready for production, your feedback will be invaluable to us. Please let us know how we can improve Miracle to best fits with your experience and needs
         
""")

# Call to action
st.header("Get Started")
st.write("""
To interact with Miracle, you will need your OpenAI API key. 

1. If you do not have an OpenAI account, please [Sign up](https://platform.openai.com/signup?launch).
2. If you already have an account, retrieve your API key from your [OpenAI Dashboard](https://platform.openai.com/account/api-keys).

Once you have your API key, proceed to the Configuration Tab to enter it.
""")

# Redirect button to Configuration page
if st.button("Proceed to Configuration"):
    switch_page("Configuration")