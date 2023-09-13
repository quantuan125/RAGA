import os
import re
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
from langchain.chains.router import MultiRetrievalQAChain



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
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = os.path.splitext(file_name)[0]
        st.write(self.file_name)
        self.reader = pypdf.PdfReader(file_path)
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
    
    def extract_pages_from_pdf(self):
        pages = []
        for page_num, page in enumerate(self.reader.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
        return pages

    def parse_pdf(self):
        """
        Extracts the title and text from each page of the PDF.
        :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
        """
        metadata = self.extract_metadata_from_pdf()
        pages = self.extract_pages_from_pdf()
        #st.write(pages)
        #st.write(metadata)
        return pages, metadata
    
    @staticmethod
    def merge_hyphenated_words(text):
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    @staticmethod
    def fix_newlines(text):
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    @staticmethod
    def remove_multiple_newlines(text):
        return re.sub(r"\n{2,}", "\n", text)

    def clean_text(self, pages):
        cleaning_functions = [
            self.merge_hyphenated_words,
            self.fix_newlines,
            self.remove_multiple_newlines,
        ]
        cleaned_pages = []
        for page_num, text in pages:
            for cleaning_function in cleaning_functions:
                text = cleaning_function(text)
            cleaned_pages.append((page_num, text))
        return cleaned_pages

    def text_to_docs(self, text):
        doc_chunks = []
        for page_num, page in text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=200,
            )
            chunks = text_splitter.split_text(page)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page_number": page_num,
                        "chunk": i,
                        "source": f"p{page_num}-{i}",
                        "file_name": self.file_name,
                        **self.metadata,
                    },
                )
                doc_chunks.append(doc)
            st.write(doc_chunks)
        return doc_chunks
    
    def get_pdf_text(self):
        pages, metadata = self.parse_pdf()  # We only need the pages from the tuple
        cleaned_text_pdf = self.clean_text(pages)
        document_chunks = self.text_to_docs(cleaned_text_pdf)
        return document_chunks

    def get_vectorstore(self):
        document_chunks = self.get_pdf_text()
        vector_stores = []
        for doc in document_chunks:
            vectorstore = FAISS.from_documents(documents=[doc], embedding=self.embeddings)
            vector_stores.append(vectorstore)
        st.write(vector_stores)
        return vector_stores
    
    def merge_stores(self, vector_stores):
        combined_store = vector_stores[0]
        st.write(combined_store)
        for store in vector_stores[1:]:
            combined_store.merge_from(store)
    
    def get_combined_vectorstore(self):
        vector_stores = self.get_vectorstore()
        self.merge_stores(vector_stores)
        st.write(vector_stores[0])
        return vector_stores[0]  # After merging, the first store in the list is the primary store.
    
    
class DatabaseTool:
    def __init__(self, llm, combined_vector_store):
         self.retrieval = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=combined_vector_store.as_retriever(),
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
        llm_search = DuckDuckGoSearchRun()
        llm_database = DatabaseTool(llm=llm, combined_vector_store=st.session_state.vector_store)

        tools = [
            Tool(
                name="Search",
                func=llm_search.run,
                description="Useful when you cannot find a clear answer by looking up the database and that you need to search the internet for information. Input should be a fully formed question based on the context of what you couldn't find and not referencing any obscure pronouns from the conversation before"
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

        PREFIX ="""Assistant is a large language model trained by OpenAI. Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
        
        Begin by searching for answers and relevant examples within PDF pages (documents) provided in the database. If you are unable to find sufficient information you may use a general internet search to find results. However, always prioritize providing answers and examples from the database before resorting to general internet search.

        If the user question does not require any tools, simply kindly respond back in an assitance manner as a Final Answer

        Assistant has access to the following tools:"""

        FORMAT_INSTRUCTIONS = """
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
            max_iterations=4
        )

        executive_agent = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=self.tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        max_iterations=4,
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
    if "doc_sources" not in st.session_state:
        st.session_state.doc_sources = []

    st.sidebar.title("Upload Local Vector DB")
    uploaded_file = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)  # You can specify the types of files you want to accept
    with st.sidebar:
        if st.button("Process"):
                with st.spinner("Processing"):
                    for file in uploaded_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                            tmpfile.write(file.getvalue())
                            temp_path = tmpfile.name
                            db_store = DBStore(temp_path, file.name)
                            combined_vector_store = db_store.get_combined_vectorstore()
                            st.session_state.vector_store = combined_vector_store
                            st.success("PDF uploaded successfully!")

    MRKL_agent = MRKL()
    #memory = load_memory(st.session_state.messages, MRKL_agent.memory)
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
        
    with st.expander("View Document Sources"):
        st.subheader("Source Document")
        if len(st.session_state.doc_sources) != 0:

            for document in st.session_state.doc_sources:
                    st.divider()
                    source_text = f"{document.page_content}\n\nDocument: {document.metadata['file_name']}\n\nPage Number: {document.metadata['page_number']}\n -Chunk: {document.metadata['chunk']}"
                    st.write(source_text)
        else:
                st.write("No document sources found")


    #st.write(memory)
    st.write(st.session_state.vector_store)



if __name__== '__main__':
    main()