import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, Tool, AgentExecutor, ConversationalAgent, ZeroShotAgent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory 
from langchain.chains import LLMMathChain
from langchain import LLMChain
import streamlit as st
from langchain.utilities import SerpAPIWrapper



def display_messages(messages):
    # Display all messages
    for msg in messages:
        st.chat_message(msg["roles"]).write(msg["content"])

def load_memory(messages, memory):
    # Skip the initial message from the assistant (index 0)
    for i in range(1, len(messages), 2):  
        user_msg = messages[i]
        if i + 1 < len(messages):  # Check if there's an assistant message after the user message
            assistant_msg = messages[i + 1]
            memory.save_context({"input": user_msg["content"]}, {"output": assistant_msg["content"]})  # Update memory with assistant's response
    return memory

class MRKL:
    def __init__(self):
        self.tools = self.load_tools()
        self.agent, self.memory = self.load_agent()

    def load_tools(self):
        # Load tools
        llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo"
            )
        llm_math = LLMMathChain(llm=llm)
        llm_search = SerpAPIWrapper()
        tools = [
            Tool(
                name="Search",
                func=llm_search.run,
                description="useful for when you need to answer questions about current events"
            ),
            Tool(
                name='Calculator',
                func=llm_math.run,
                description='Useful for when you need to answer questions about math.'
            )
        ]
        return tools

    def load_agent(self):
        llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo"
            )

        PREFIX = """Assistant is a large language model trained by OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

        TOOLS:
        ------

        Assistant has access to the following tools:"""

        FORMAT_INSTRUCTIONS = """
        To answer a question or respond to a statement, follow these steps:

        1. Determine if the input requires the use of a tool. If it does not, then provide a kind response and skip right to the final answer.  
        2. If a tool is necessary, identify the information you need. If it's not in your current knowledge, use the Search tool to fetch it.
        3. Extract the necessary information from the Search tool. 
        4. If a mathematical operation is required, ensure you have all numerical values needed.
        5. Use the Calculator tool with the extracted numerical values to perform the calculation.
        6. Provide the final answer.

        For example: 
        If you need to find out someone's age and then perform a calculation with it (For example: What is Robert F Kennedy Jr's age and subtract by 10 and multiply by 21?)
        - First, use the Search tool to find the age. For example: "I need to find Robert F Kennedy Jr's age first."
        - Extract the age from the search results. For example: "Robert F Kennedy Jr's age is 69"
        - Use the Calculator tool with the extracted age to perform the calculation. For example: "(69-10)*21"
        - Provide the final answer. For example: "The answer is 1239"

        Use the following format:
        '''
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}] or if no tool is needed, then skip to Final Thought
        Action Input: the input to the action. 
        Observation: the result of the action 

        If your Thought mentions multiple tools, your next Action should correspond to the first tool you mentioned.
        If your Thought does not mentions any tools, skip to the Final Thought

        ... (this Thought/Action/Action Input/Observation can repeat N times)

        Final Thought: I now know the final answer
        Final Answer: the final answer to the original input question.
        '''
        """
        

        SUFFIX = """Begin!
        Previous conversation history:{chat_history}
        Question: {input}
        {agent_scratchpad}"""


        prompt = ZeroShotAgent.create_prompt(
            self.tools, 
            prefix=PREFIX,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            input_variables=["chat_history", "input", "agent_scratchpad"])
        
        def _handle_error(error) -> str:
            """If you encounter a parsing error:
            1. Review the tool's output and ensure you've extracted the necessary information.
            2. Ensure you're using the correct format for the next step.
            3. If you're unsure about the next step, refer back to the format instructions.
            4. If all else fails, restart the process and try again."""
            return str(error)[:50]

        memory = ConversationBufferMemory(memory_key="chat_history")

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            tools=self.tools,
            llm=llm,
            handle_parsing_errors=True,
            max_iterations=7
        )

        executive_agent = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=self.tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        max_iterations=7,
        )

        return executive_agent, memory
    
    def run_agent(self, input, callbacks=[]):
        # Define the logic for processing the user's input
        # For now, let's just use the agent's run method
        response = self.agent.run(input=input, callbacks=callbacks)
        return response



def main():
    load_dotenv()

    st.set_page_config(page_title="MRKL AGENT", page_icon="🦜️", layout="wide")
    st.title("🦜️ MRKL AGENT")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"roles": "assistant", "content": "How can I help you?"}]
    if "user_input" not in st.session_state:
        st.session_state.user_input = None
     
    openai_api_key = os.getenv("OPENAI_API_KEY")

    #st.sidebar.title("Upload Local Vector DB")
    #uploaded_file = st.sidebar.file_uploader("Choose a file")  # You can specify the types of files you want to accept
    #if uploaded_file:
        #st.sidebar.write("File uploaded successfully!")

    MRKL_agent = MRKL()
    memory = load_memory(st.session_state.messages, MRKL_agent.memory)
    display_messages(st.session_state.messages)

    if user_input := st.chat_input("Type something here..."):
        st.session_state.user_input = user_input
        st.session_state.messages.append({"roles": "user", "content": st.session_state.user_input})
        st.chat_message("user").write(st.session_state.user_input)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = MRKL_agent.run_agent(input=st.session_state.user_input, callbacks=[st_callback])
            st.session_state.messages.append({"roles": "assistant", "content": response})
            st.write(response)

    st.write(memory)

if __name__== '__main__':
    main()