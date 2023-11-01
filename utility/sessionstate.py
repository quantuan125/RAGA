import streamlit as st
from agent.miracle import MRKL
from utility.client import ClientDB
import pinecone
import os

class Init:
    def initialize_session_state():
        default_values = {
            # Common to agent.py and chat.py
            "llm_model": "gpt-3.5-turbo",
            "use_retriever_model": False,
            "vector_store": None,
            "br18_exp": False,
            "web_search": False,
            "system_message_content": """
            You are Miracle, an expert in construction, legal frameworks, and regulatory matters.

                You have the following tools to answer user queries, but only use them if necessary. 
                """,  
            "formatting_message_content": """
            Your primary objective is to provide responses that:
                1. Offer an overview of the topic, referencing the chapter and the section if relevant.
                2. List key points in bullet-points or numbered list format, referencing the clauses and their respective subclauses if relevant.
                3. Always match or exceed the details of the tool's output text in your answers. 
                4. Reflect back to the user's question and give a concise conclusion.
                5. If there is not enough information or no clear answer, explicitly state so. 
                """,  
            "reflection_message_content": """
            Reminder: 
                Always try all your tools to find the answer to the user query

                Always self-reflect your answer based on the user's query and follows the list of response objective. 
                """,  
            "selected_collection_state": None,

            # Specific to chat.py
            "messages": [{"roles": "assistant", "content": "Hi, I am Miracle. How can I help you?"}],
            "user_input": None,
            "summary": None,
            "doc_sources": [],
            "br18_vectorstore": None,
            "history": None,
            "token_count": 0,
            "focused_mode": False,
            "selected_document": None,
            "s3_object_url": None,

            # Specific to dbm.py
            "delete": False,
            "show_info": False,

            # Specific to configuration.py
            "username": None,
            "authentication": False,

            # Specific to app.py
            "openai_key": ""
        }

        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def initialize_agent_state():
        if "agent" not in st.session_state:
            st.session_state.agent = MRKL()

    def initialize_clientdb_state():
        if st.session_state.username is None:
            st.warning("Please sign in through the configuration tab before proceeding.")
        elif 'client_db' not in st.session_state:
            st.session_state.client_db = ClientDB(username=st.session_state.username, collection_name=None, load_vector_store=False)

    def initialize_pinecone_state():
            pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
            )