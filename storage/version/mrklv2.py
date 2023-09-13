import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory 
from langchain.chains import LLMMathChain

import streamlit as st



def handle_messages_display_and_memory(messages, memory):
    # Display the initial "assistant" message
    initial_msg = messages[0]
    st.chat_message(initial_msg["roles"]).write(initial_msg["content"])
    
    # Process the remaining messages starting from index 1
    for i in range(1, len(messages), 2):  
        user_msg = messages[i]
        st.chat_message(user_msg["roles"]).write(user_msg["content"])

        if i + 1 < len(messages):  # Check if there's an assistant message after the user message
            assistant_msg = messages[i + 1]
            st.chat_message(assistant_msg["roles"]).write(assistant_msg["content"])
            memory.save_context({"input": user_msg["content"]}, {"output": assistant_msg["content"]})  # Update memory with assistant's response

    return memory

def load_conversation_agent():
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm = OpenAI(temperature=0.5, streaming=True)

    llm_math = LLMMathChain(llm=llm)
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
        ),

        Tool(
        name='Calculator',
        func=llm_math.run,
        description='Useful for when you need to answer questions about math.'
        )
            ]
    
    conversation_agent = initialize_agent(
        #tools=tools,
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  
        handle_parsing_errors=True,
        memory=memory 
        )

    return conversation_agent, memory


def main():
    load_dotenv()

    st.set_page_config(page_title="MRKL AGENT", page_icon="🦜️", layout="wide")
    st.title("🦜️ MRKL AGENT")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"roles": "assistant", "content": "How can I help you?"}]
    if "prompt" not in st.session_state:
        st.session_state.prompt = None
     
    openai_api_key = os.getenv("OPENAI_API_KEY")

    st.sidebar.title("Upload Local Vector DB")
    uploaded_file = st.sidebar.file_uploader("Choose a file")  # You can specify the types of files you want to accept
    if uploaded_file:
        st.sidebar.write("File uploaded successfully!")

    conversation_agent, memory = load_conversation_agent()
    memory = handle_messages_display_and_memory(st.session_state.messages, conversation_agent.memory)

    if prompt := st.chat_input("Type something here..."):
        st.session_state.prompt = prompt
        st.session_state.messages.append({"roles": "user", "content": st.session_state.prompt})
        st.chat_message("user").write(st.session_state.prompt)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = conversation_agent.run(input=st.session_state.prompt, callbacks=[st_callback])
            st.session_state.messages.append({"roles": "assistant", "content": response})
            st.write(response)

    st.write(memory)

if __name__== '__main__':
    main()