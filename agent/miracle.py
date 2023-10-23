import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import streamlit as st
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import Tool, AgentExecutor
from langchain.schema.messages import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.prompts import MessagesPlaceholder
from .tools import BR18_DB, DatabaseTool, CustomGoogleSearchAPIWrapper

class MRKL:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0, 
            streaming=st.session_state.get('streaming', True),
            model_name=st.session_state.llm_model,
            max_tokens=500
            )
        self.tools = self.load_tools()
        self.agent_executor, self.memory = self.load_agent()

    def conversational_tool_func(*args, **kwargs):
        return "Conversational skills activated. No action performed."

    def load_tools(self):
        current_directory = os.getcwd()
        # Load tools
        tools = []
        
        tools.append(
            Tool(
                name='Conversational_Tool',
                func=MRKL.conversational_tool_func,
                description='Using your conversational skills as an assistance to answer user queries and concerns'
            )
        )

        llm_search = CustomGoogleSearchAPIWrapper()
    
        existing_tool = next((tool for tool in tools if tool.name == 'Google_Search'), None)
        
        if st.session_state.web_search:
            if existing_tool:
                existing_tool.func = llm_search.run
            else:
                tools.append(
                    Tool(
                        name="Google_Search",
                        func=llm_search.run,
                        description="Useful for web search."
                    )
                )
        else:
            if existing_tool:
                existing_tool.func = llm_search.disabled_function

        if st.session_state.vector_store is not None:
            selected_document = getattr(st.session_state, 'selected_document', None)
            metadata = getattr(st.session_state, 'document_metadata', None)
            file_name = getattr(st.session_state, 'document_filename', None)    
            vector_store = st.session_state.vector_store
            llm_database = DatabaseTool(
                llm=self.llm, 
                vector_store=vector_store, 
                metadata=metadata, 
                filename=file_name,
                selected_document=selected_document)

            tools.append(
                Tool(
                    name='Document_Database',
                    func=llm_database.run,
                    description=llm_database.get_description(),
                ),
            )

        #st.write(llm_database.get_description())

        if st.session_state.br18_exp is True:
            br18_folder_path = os.path.join(current_directory, "BR18_DB")
            llm_br18 = BR18_DB(llm=self.llm, folder_path=br18_folder_path)

            tools.append(
            Tool(
                name='BR18_Database',
                func=llm_br18.run,
                description="""
                Always useful for when you need to answer questions about the Danish Building Regulation 18 (BR18). 
                Input should be the specific keywords from the user query. Exclude the following common terms and their variations or synonyms especially words such as 'building' and 'regulation'.
                Use this tool more often than the normal search tool.
                """
            )
            )

        return tools

    def load_agent(self):
        
        # Memory
        max_token_limit = st.session_state.get('max_token_limit', 1300)
        chat_msg = StreamlitChatMessageHistory(key="mrkl_chat_history")
        memory_key = "history"
        memory = AgentTokenBufferMemory(memory_key=memory_key, llm=self.llm, input_key='input', output_key="output", max_token_limit=max_token_limit, chat_memory=chat_msg)
        st.session_state.history = memory

        # System Message
        system_message = SystemMessage(content=st.session_state.system_message_content)

        formatting_message = SystemMessage(content=st.session_state.formatting_message_content)

        if st.session_state.reflection_message_content:
            reflection_message = SystemMessage(content=st.session_state.reflection_message_content)
        else:
            reflection_message = None

        # Prompt
        if reflection_message:
            prompt = OpenAIFunctionsAgent.create_prompt(
                system_message=system_message,
                extra_prompt_messages=[formatting_message, MessagesPlaceholder(variable_name=memory_key), reflection_message]
            )
        else:
            prompt = OpenAIFunctionsAgent.create_prompt(
                system_message=system_message,
                extra_prompt_messages=[formatting_message, MessagesPlaceholder(variable_name=memory_key)]
            )

        # Agent
        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)
        
        # Agent Executor
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, memory=memory, verbose=True, return_intermediate_steps=True)
        
        return agent_executor, memory

    def clear_conversation(self):
        self.memory.clear()

    def run_agent(self, input, callbacks=[]):
        with get_openai_callback() as cb:
            result = self.agent_executor({"input": input}, callbacks=callbacks)
            st.session_state.token_count = cb
            print(cb)
        return result