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
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
import tempfile
import pypdf
import json
import openai
from langchain.docstore.document import Document



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

class DBStore:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.reader = pypdf.PdfReader(pdf_file)
        self.metadata = self.extract_metadata_from_pdf()
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def extract_metadata_from_pdf(self):
        """Extract metadata from the PDF."""
        metadata = self.reader.metadata
        return {
            "title": metadata.get("/Title", "").strip(),
            "author": metadata.get("/Author", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }

    def get_pdf_text(self):
        loaders = [PyPDFLoader(self.pdf_file)]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return docs  # Return documents instead of plain text

    def get_text_chunks(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_documents(documents)

    def get_vectorstore(self):
        documents = self.get_pdf_text()
        chunks = self.get_text_chunks(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        return vectorstore

class DatabaseTool:
    def __init__(self, llm, vector_store):
        self.retrieval = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

    def run(self, query: str):
        output = self.retrieval(query)
        st.session_state.doc_sources = output['source_documents']
        return output['result']


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
        llm_database = DatabaseTool(llm=llm, vector_store=st.session_state.vector_store)

        tools = [
            Tool(
                name="Search",
                func=llm_search.run,
                description="Useful when you cannot find a clear answer by looking up the database and that you need to search the regular internet for information. You can also this tool to find general information and about current events"
            ),
            Tool(
                name='Calculator',
                func=llm_math.run,
                description='Useful for when you need to answer questions about math.'
            ),
            Tool(
                name='Look up database',
                func=llm_database.run,
                description="Always useful for finding the exactly written answer to the question by looking into a collection of documents. Input should be a query, not referencing any obscure pronouns from the conversation before that will pull out relevant information from the database."
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

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
        
        Begin by searching for answers and relevant examples within the database provided. If you are unable to find sufficient information you may use a general internet search to find results. 
        
        However, always prioritize providing answers and examples from the database before resorting to general internet search.

        Assistant has access to the following tools:"""

        FORMAT_INSTRUCTIONS = """
        To answer a question or respond to a statement, follow these steps:

        1. Determine if the input requires the use of a tool. If it does not, then provide a kind response and skip right to the final answer.  
        2. If a tool is necessary, identify the information you need. If it's not in your current knowledge, use the Search tool to fetch it.
        3. Extract the necessary information from the Search tool. 
        4. If a mathematical operation is required, ensure you have all numerical values needed.
        5. Use the Calculator tool with the extracted numerical values to perform the calculation.
        6. Provide the final answer.

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
            max_iterations=5
        )

        executive_agent = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=self.tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        max_iterations=5,
        )

        return executive_agent, memory
    
    def get_keywords(self, llm_response):
        conversation = llm_response["chat_history"]
        keyword_list = []

        search_keywords_extract_function = {
            "name": "search_keywords_extractor",
            "description": "Creates a list of 5 short academic Google searchable keywords from the given conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "List of 5 short academic Google searchable keywords"
                    }
                },
                "required": ["keywords"]
            }
        }

        res = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=[{"role": "user", "content": conversation}],
            functions=[search_keywords_extract_function]
        )

        if "function_call" in res['choices'][0]['message']:
            args = json.loads(res['choices'][0]['message']['function_call']['arguments'])
            keyword_list = list(args['keywords'].split(","))

        return keyword_list

    
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
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
     
    openai_api_key = os.getenv("OPENAI_API_KEY")

    st.sidebar.title("Upload Local Vector DB")
    uploaded_file = st.sidebar.file_uploader("Choose a file")  # You can specify the types of files you want to accept
    with st.sidebar:
        if st.button("Process"):
                with st.spinner("Processing"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                        tmpfile.write(uploaded_file.getvalue())
                        temp_path = tmpfile.name
                        db_store = DBStore(temp_path)
                        vector_store = db_store.get_vectorstore()
                        # You can save the vector store and any other data to the session state if needed
                        st.session_state.vector_store = vector_store
                        st.success("PDF uploaded successfully!")
                        metadata = db_store.extract_metadata_from_pdf()
                        st.write("PDF Metadata:", metadata)

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
    #st.write(st.session_state.messages)


if __name__== '__main__':
    main()