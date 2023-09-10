import os
import re
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, Tool, AgentExecutor, ConversationalAgent, ZeroShotAgent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import LLMMathChain
from langchain.chains import LLMChain, StuffDocumentsChain
import streamlit as st
import langchain
from langchain.utilities import SerpAPIWrapper
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
import tempfile
import pypdf
import json
import openai
from pathlib import Path
from langchain.docstore.document import Document
from langchain.chains.router import MultiRetrievalQAChain
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, WebBaseLoader, UnstructuredMarkdownLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import lark
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
import  streamlit_toggle as tog
import langchain
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Set
import pinecone
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from typing import List, Dict, Any 
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.question_answering import map_reduce_prompt, stuff_prompt
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage, BaseMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid

langchain.debug = True
langchain.verbose = True

def on_selectbox_change():
    st.session_state.show_info = True

def reset_chat():
    st.session_state.messages = [{"roles": "assistant", "content": "Hi, I am Miracle. How can I help you?"}]
    st.session_state.history = []
    st.session_state.search_keywords = []
    st.session_state.doc_sources = []
    st.session_state.history = None
    st.session_state.summary = None
    st.session_state.agent.clear_conversation()
    st.session_state.vector_store = None

def display_messages(messages):
    # Display all messages
    for msg in messages:
        st.chat_message(msg["roles"]).write(msg["content"])

class DBStore:
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = os.path.splitext(file_name)[0]
        self.reader = pypdf.PdfReader(file_path)
        self.metadata = self.extract_metadata_from_pdf()
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def extract_metadata_from_pdf(self):
        """Extract metadata from the PDF."""
        metadata = self.reader.metadata
        st.session_state.document_metadata = metadata
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
            #st.write(doc_chunks)
        return doc_chunks
    
    def get_pdf_text(self):
        pages, metadata = self.parse_pdf()  # We only need the pages from the tuple
        cleaned_text_pdf = self.clean_text(pages)
        document_chunks = self.text_to_docs(cleaned_text_pdf)
        return document_chunks

    def get_vectorstore(self):
        document_chunks = self.get_pdf_text()
        vector_store = FAISS.from_documents(documents=document_chunks, embedding=self.embeddings)
        #st.write(vector_store)
        return vector_store
   
class DatabaseTool:
    def __init__(self, llm, vector_store, metadata=None):
        self.retrieval = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        self.metadata = metadata

    def get_description(self):
        base_description = "Always useful for finding the exactly written answer to the question by looking into a collection of documents."
        if self.metadata:
            title = self.metadata.get("/Title")
            author = self.metadata.get("/Author")
            subject = self.metadata.get("/Subject")
            if author:
                return f"{base_description} This tool is currently loaded with '{title}' by '{author}', and has a topic of '{subject}'. Input should be a query, not referencing any obscure pronouns from the conversation before that will pull out relevant information from the database. Use this more than the normal search tool"
            else:
                return f"{base_description} This tool is currently loaded with '{title}', and has a topic of {subject}. Input should be a query, not referencing any obscure pronouns from the conversation before that will pull out relevant information from the database. Use this more than the normal search tool"
        return base_description


    def run(self, query: str):
        output = self.retrieval(query)
        st.session_state.doc_sources = output['source_documents']
        return output['result']

class BR18_DB:
    def __init__(self, llm, folder_path: str):
        self.llm = llm
        self.folder_path = folder_path
        self.md_paths = self.load_documents()  # Renamed from pdf_paths to md_paths
        self.embeddings = OpenAIEmbeddings()
        self.pinecone_index_name = "br18"
        if self.pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(self.pinecone_index_name, dimension=1536)
            self.vectorstore = self.create_vectorstore()
        else:
            self.vectorstore = Pinecone.from_existing_index(self.pinecone_index_name, self.embeddings)
        self.retriever = None

    def load_documents(self):
        md_paths = list(Path(self.folder_path).rglob("*.md"))
        documents = []
        for path in md_paths:
            loader = TextLoader(str(path))
            #st.write(loader)
            data = loader.load()
            documents.extend(data)  # Assuming data is a list of Document objects
        #st.text(documents)
        return documents
    
    def split_and_chunk_text(self, markdown_document: Document):
        # Extract the markdown text from the Document object
        markdown_text = markdown_document.page_content
        #st.write(f"Type of markdown_document: {type(markdown_document)}")
        #st.markdown(markdown_text)
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        # Initialize MarkdownHeaderTextSplitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        #st.write(markdown_splitter)
        
        # Split the document by headers
        md_header_splits = markdown_splitter.split_text(markdown_text)
        #st.write(md_header_splits)
        #st.write(type(md_header_splits[0]))
        
        # Define chunk size and overlap
        chunk_size = 1500
        chunk_overlap = 200
        
        # Initialize RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        
        # Split the header-split documents into chunks
        all_splits = text_splitter.split_documents(md_header_splits)
        #st.write(all_splits)
        # Output for debugging

        return all_splits
    
    def process_all_documents(self):
        all_processed_splits = []
        for markdown_document in self.md_paths:
            processed_splits = self.split_and_chunk_text(markdown_document)
            all_processed_splits.extend(processed_splits)
        #st.write(all_processed_splits)
        return all_processed_splits
    
    def create_vectorstore(self):
        # Use the process_all_documents method to get all the processed splits
        all_splits = self.process_all_documents()
        #st.write(all_splits)



        # Create and save the vector store to cloud
        br18_vectorstore = Pinecone.from_documents(documents=all_splits, embedding=self.embeddings, index_name=self.pinecone_index_name)

        # Store the vector store in the session state
        st.session_state.br18_vectorstore = br18_vectorstore
        

        return br18_vectorstore
    
    def get_keywords(self, query: str) -> list:
        # Define the prompt template
        prompt_template = """
        The user is searching for specific information within a set of documents about building regulation.
        Your task is to list only the specific keywords from the following user query: {query}

        Please adhere to the following guidelines: 
        - Interpret intent and correct typos.
        - Include both singular and plural forms.
        - Consider formal hyphenations and compound words.
        Exclude the following common terms and their variations or synonyms especially for these words: 
        - "building", "buildings", "construction", "constructions"
        - "regulation", "regulations", "requirement", "regulatory", "requirements"


        Example 1: 
        Query: "What is the building regulation regarding stairs"
        Answer: "stair, stairs"

        Example 2: 
        Query: "How is the law applied to ventilation in residential building?"
        Answer: "ventionlation, residential"

        Example 3: 
        Query: "List the regulatory requirement regarding fire safety"
        Answer: "fire safety"

        Example 4:
        Query: "What are the regulations regarding noise in non residential building?"
        Answer: "noise, non-residential"
        """

        llm_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )
        # Extract keywords from the query
        keywords_str = llm_chain.predict(query=query)
        # Convert the keywords string into a list
        keywords = keywords_str.split(", ")

        return keywords

    def post_filter_documents(self, keywords: List[str], source_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered_docs = []
        header_4_matched_docs = []
        header_3_matched_docs = []

        for doc in source_docs:
            headers = doc.metadata
            #st.write(f"Current Document Metadata: {headers}")

            # Check if Header 4 exists
            header_4_exists = 'Header 4' in headers
            #st.write(f"Does Header 4 Exist?: {header_4_exists}")

            header_3_content = headers.get('Header 3', "").lower()
            header_4_content = headers.get('Header 4', "").lower()

            #st.write(f"Header 3 Content: {header_3_content}")
            #st.write(f"Header 4 Content: {header_4_content}")

            header_3_matched_keywords = set()
            header_4_matched_keywords = set()

            for keyword in keywords:
                keyword_lower = keyword.lower()
                #st.write(f"Checking Keyword: {keyword_lower}")

                # Check for matches in Header 4 only if it exists
                if header_4_exists:
                    if keyword_lower in header_4_content:
                        #st.write("Keyword matches in Header 4.")
                        header_4_matched_keywords.add(keyword_lower)
                else:
                    # If Header 4 doesn't exist, check for matches in Header 3
                    if keyword_lower in header_3_content:
                        #st.write("Keyword matches in Header 3.")
                        header_3_matched_keywords.add(keyword_lower)

            #st.write(f"Matched Keywords in Header 3: {header_3_matched_keywords}")
            #st.write(f"Matched Keywords in Header 4: {header_4_matched_keywords}")

            # Determine whether to add the document to the filtered list
            if header_4_exists and header_4_matched_keywords:
                #st.write("Document added based on Header 4 matches.")
                header_4_matched_docs.append(doc)
            elif not header_4_exists and header_3_matched_keywords:
                #st.write("Document added based on Header 3 matches.")
                header_3_matched_docs.append(doc)

            filtered_docs = header_4_matched_docs + header_3_matched_docs

        return filtered_docs

    def create_retriever(self, query: str):
        retriever = self.vectorstore.as_retriever(
        search_kwargs={'k': 20}
        )

        relevant_docs = retriever.get_relevant_documents(query)
        #st.write(relevant_docs)
        #st.write(type(relevant_docs))

        keywords = self.get_keywords(query)
        #st.write(keywords)


        filtered_docs = self.post_filter_documents(keywords, relevant_docs)
        #st.write(filtered_docs)
        #st.write(type(filtered_docs[0]))

        if not filtered_docs:
            non_filtered = relevant_docs[:5]
            #st.write(non_filtered)
            return non_filtered

        return filtered_docs
    
    def run(self, query: str):
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        The answer should be as specific as possible and reference clause numbers and their respective subclause. 
        Make sure to mention requirement numbers and specific integer values where relevant.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Retrieve the filtered documents
        filtered_docs = self.create_retriever(query)
        #st.write(type(filtered_docs[0]))
        #st.write(filtered_docs)

        qa_chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        output = qa_chain({"input_documents": filtered_docs, "question": query}, return_only_outputs=True)

        st.session_state.doc_sources = filtered_docs

        return output


class BR18_Appendix: 
    def __init__(self, llm, folder_path: str):
        self.llm = llm
        self.folder_path = folder_path
        self.md_paths = self.load_documents()  # Renamed from pdf_paths to md_paths
        self.embeddings = OpenAIEmbeddings()
        self.retriever = self.create_vectorstore()

    def load_documents(self):
        md_paths = list(Path(self.folder_path).rglob("*.md"))
        documents = []
        for path in md_paths:
            loader = TextLoader(str(path))
            data = loader.load()
            documents.extend(data)  # Assuming data is a list of Document objects
        #st.text(documents)
        return documents
    
    def split_and_chunk_text(self, markdown_document: Document):
        # Extract the markdown text from the Document object
        markdown_text = markdown_document.page_content
        #st.write(f"Type of markdown_document: {type(markdown_document)}")
        #st.markdown(markdown_text)
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        # Initialize MarkdownHeaderTextSplitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        #st.write(markdown_splitter)
        
        # Split the document by headers
        md_header_splits = markdown_splitter.split_text(markdown_text)
        #st.write(md_header_splits)
        #st.write(type(md_header_splits[0]))
        
        # Define chunk size and overlap
        parent_chunk_size = 3500
        parent_chunk_overlap = 0
        
        # Initialize RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
        )
        
        # Split the header-split documents into chunks
        all_parent_splits = text_splitter.split_documents(md_header_splits)

        for split in all_parent_splits:
            metadata_str = f"{split.metadata.get('Header 3', '')}\n\n{split.metadata.get('Header 4', '')}"
            split.page_content = f"{metadata_str}\n\n{split.page_content}"

        return all_parent_splits
    
    def generate_child_splits(self, parent_splits: List[Document]) -> List[Document]:
        child_chunk_size = 300

        child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=0
        )

        all_child_splits = []
        for parent_split in parent_splits:
            child_splits = child_text_splitter.split_text(parent_split.page_content)
            first_child_split = Document(
            page_content=child_splits[0],  # Assuming this is a string
            metadata=parent_split.metadata  # You can copy the metadata from the parent or set as needed
            )
            all_child_splits.append(first_child_split)  # Append only the first child split (assuming it contains the metadata)


        return all_child_splits

    def process_all_documents(self):
        all_parent_splits = []  # Local variable to store all parent splits
        all_child_splits = []  # Local variable to store all child splits
        
        for markdown_document in self.md_paths:
            parent_splits = self.split_and_chunk_text(markdown_document)
            child_splits = self.generate_child_splits(parent_splits)
            
            all_parent_splits.extend(parent_splits)
            all_child_splits.extend(child_splits)

        st.write(all_parent_splits)
        st.write(all_child_splits)

        return all_parent_splits, all_child_splits  # Return both lists
    
    def create_vectorstore(self):
        all_parent_splits, all_child_splits = self.process_all_documents()
        
        id_key = "doc_id" 
        br18_appendix_parent_store = InMemoryStore()
        parent_doc_ids = [str(uuid.uuid4()) for _ in all_parent_splits]
        br18_appendix_parent_store.mset(list(zip(parent_doc_ids, all_parent_splits)))

        for parent_id, child_split in zip(parent_doc_ids, all_child_splits):
            child_split.metadata[id_key] = parent_id

        # Create and save the vector store to disk
        br18_appendix_child_vectorstore = FAISS.from_documents(documents=all_child_splits, embedding=self.embeddings)
        st.write(br18_appendix_child_vectorstore)

        # Store the vector store in the session state
        st.session_state.br18_appendix_vectorstore = br18_appendix_child_vectorstore

        retriever = MultiVectorRetriever(
        vectorstore=br18_appendix_child_vectorstore, 
        docstore=br18_appendix_parent_store,  # Use the in-memory parent store
        id_key=id_key  # The key used to match parent and child documents
        )
        st.write(retriever)
        st.write(all_child_splits)
        
        return retriever
    
    def run(self, query: str):
        relevant_child_docs = self.retriever.vectorstore.similarity_search(query)
        st.write(relevant_child_docs)
        parent_docs = self.retriever.get_relevant_documents(query)[:1]
        st.write(parent_docs)

        prompt_template = """Use the following tables from the BR18 appendix to answer the question at the end. 
        The answer should be as specific as possible in terms of requirement numbers and value, referencing the appendix and table numbers where the answer originates.  
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )


        qa_chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        output = qa_chain({"input_documents": parent_docs, "question": query}, return_only_outputs=True)

        st.session_state.doc_sources = parent_docs

        return output


class SummarizationTool():
    def __init__(self, document_chunks):
        self.llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo"
        )
        self.document_chunks = document_chunks
        self.map_prompt_template, self.combine_prompt_template = self.load_prompts()
        self.chain = self.load_summarize_chain()

    def load_prompts(self):
        map_prompt = '''
        Summarize the following text in a clear and concise way:
        TEXT:`{text}`
        Brief Summary:
        '''
        combine_prompt = '''
        Generate a summary of the following text that includes the following elements:

        * A title that accurately reflects the content of the text.
        * An introduction paragraph that provides an overview of the topic.
        * Bullet points that list the key points of the text.

        Text:`{text}`
        '''
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        return map_prompt_template, combine_prompt_template
    
    def load_summarize_chain(self):
        return load_summarize_chain(
            llm=self.llm,
            chain_type='map_reduce',
            map_prompt=self.map_prompt_template,
            combine_prompt=self.combine_prompt_template,
            verbose=True
        )

    def run(self, query=None):
        return self.run_chain()

    def run_chain(self):
        return self.chain.run(self.document_chunks)

class MRKL:
    def __init__(self):
        self.tools = self.load_tools()
        self.agent_executor, self.memory = self.load_agent()

    def load_tools(self):
        # Load tools
        llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo"
            )
        llm_math = LLMMathChain(llm=llm)
        llm_search = SerpAPIWrapper()

        current_directory = os.getcwd()

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
        ]

        if st.session_state.vector_store is not None:
            metadata = st.session_state.document_metadata
            llm_database = DatabaseTool(llm=llm, vector_store=st.session_state.vector_store, metadata=metadata)

            #st.write(llm_database.get_description())

            tools.append(
                Tool(
                    name='Document_Database',
                    func=llm_database.run,
                    description=llm_database.get_description(),
                ),
            )

        if st.session_state.br18_exp is True:
            br18_folder_path = os.path.join(current_directory, "BR18_DB")
            llm_br18 = BR18_DB(llm=llm, folder_path=br18_folder_path)

            appendix_folder_path = os.path.join(current_directory, "BR18_Appendix")
            llm_br18_appendix = BR18_Appendix(llm=llm, folder_path=appendix_folder_path)

            tools.extend([
            Tool(
                name='BR18_Database',
                func=llm_br18.run,
                description="Always useful for when you need to answer questions about the Danish Building Regulation 18 (BR18). Input should be a fully formed question. Use this tool more often than the normal search tool"
            ),
            Tool(
                name='BR18_Appendix',
                func=llm_br18_appendix.run,
                description="Specialized tool for answering questions related to the appendix section of the Danish Building Regulation 18 (BR18)."
            )
            ])
        return tools

    def load_agent(self):
        llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo"
            )
        
        # Memory
        memory_key = "history"
        memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, input_key='input', output_key="output", max_token_limit=8000)
        st.session_state.history = memory

        system_message_content = """
        You are MRKL, an expert in construction, legal frameworks, and regulatory matters.
        
        You are designed to be an AI Chatbot for the engineering firm COWI, and you have the following tools to answer user queries, but only use them if necessary.

        Unless otherwise explicitly stated, the user queries are about the context given.

        Your primary objective is to provide responses that:
        1. Offer an overview of the topic.
        2. List key points or clauses in a bullet-point or numbered list format.
        3. Reflect back to the user's question and give a concise conclusion.
        
        You must maintain a professional and helpful demeanor in all interactions.
        """

        # System Message
        system_message = SystemMessage(content=system_message_content)

        reflection_message_content = """
        Reminder: Always self-reflect your answer based on the user's query and follows the list of primary objective. 
        """

        reflection_message = SystemMessage(content=reflection_message_content)

        # Prompt
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key), reflection_message]
        )

        
        # Agent
        agent = OpenAIFunctionsAgent(llm=llm, tools=self.tools, prompt=prompt)
        
        # Agent Executor
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, memory=memory, verbose=True, return_intermediate_steps=True)
        
        return agent_executor, memory

    def clear_conversation(self):
        self.memory.clear()

    def run_agent(self, input, callbacks=[]):
        # Define the logic for processing the user's input
        # For now, let's just use the agent's run method
        result = self.agent_executor({"input": input}, callbacks=callbacks)
        return result
    

def main():
    load_dotenv()
    pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
)

    st.set_page_config(page_title="MRKL AGENT", page_icon="🦜️", layout="wide")
    st.title("🦜️ MRKL AGENT")

    if 'openai' not in st.session_state:
        st.session_state.openai = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{"roles": "assistant", "content": "Hi, I am Miracle. How can I help you?"}]
    if "user_input" not in st.session_state:
        st.session_state.user_input = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "doc_sources" not in st.session_state:
        st.session_state.doc_sources = []
    if "br18_vectorstore" not in st.session_state:
        st.session_state.br18_vectorstore = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'br18_exp' not in st.session_state:
        st.session_state.br18_exp = False
    if 'chat_exp' not in st.session_state:
        st.session_state.chat_exp = False

    if "agent" not in st.session_state:
        st.session_state.agent = MRKL()
    if 'show_info' not in st.session_state:
        st.session_state.show_info = False

    with st.expander("Configuration", expanded = False):
        openai_api_key = st.text_input("Enter OpenAI API Key", value="", placeholder="Enter the OpenAI API key which begins with sk-", type="password")
        if openai_api_key:
            st.session_state.openai = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.write("API key has entered")

    with st.sidebar:


        br18_experiment = st.checkbox("Experimental Feature: Enable BR18", value=False)
        if br18_experiment != st.session_state.br18_exp:
            st.session_state.br18_exp = br18_experiment
            st.session_state.agent = MRKL()

        st.sidebar.title("Upload Document to Database")
        uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)  # You can specify the types of files you want to accept
        if uploaded_files:
            file_details = {"FileName": [], "FileType": [], "FileSize": []}

            # Populate file_details using traditional loops
            for file in uploaded_files:
                file_details["FileName"].append(file.name)
                file_details["FileType"].append(file.type)
                file_details["FileSize"].append(file.size)

            # Use selectbox to choose a file
            selected_file_name = st.sidebar.selectbox('Choose a file:', file_details["FileName"], on_change=on_selectbox_change)

            # Get the index of the file selected
            file_index = file_details["FileName"].index(selected_file_name)

            # Display details of the selected file
            st.sidebar.write("You selected:")
            st.sidebar.write("FileName : ", file_details["FileName"][file_index])
            st.sidebar.write("FileType : ", file_details["FileType"][file_index])
            st.sidebar.write("FileSize : ", file_details["FileSize"][file_index])

            # Add a note to remind the user to press the "Process" button
            if st.session_state.show_info:
                st.sidebar.info("**Note:** Remember to press the 'Process' button for the current selection.")
                st.session_state.show_info = False

            with st.sidebar:
                if st.sidebar.button("Process"):
                    with st.spinner("Processing"):
                        selected_file = uploaded_files[file_index]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                            tmpfile.write(selected_file.getvalue())
                            temp_path = tmpfile.name
                            db_store = DBStore(temp_path, selected_file.name)

                            document_chunks = db_store.get_pdf_text()
                            st.session_state.document_chunks = document_chunks
                            #st.write(document_chunks)

                            vector_store = db_store.get_vectorstore()
                            st.session_state.vector_store = vector_store
                            st.session_state.agent = MRKL()
                            #st.write(st.session_state.document_metadata)
                            st.success("PDF uploaded successfully!")

                if "document_chunks" in st.session_state:
                        if st.sidebar.button("Create Summary"):
                            with st.spinner("Summarizing"):
                                summarization_tool = SummarizationTool(document_chunks=st.session_state.document_chunks)
                                st.session_state.summary = summarization_tool.run()
                                # Append the summary to the chat messages
                                st.session_state.messages.append({"roles": "assistant", "content": st.session_state.summary})
        else:
                st.session_state.vector_store = None


    display_messages(st.session_state.messages)

    if user_input := st.chat_input("Type something here..."):
        st.session_state.user_input = user_input
        st.session_state.messages.append({"roles": "user", "content": st.session_state.user_input})
        st.chat_message("user").write(st.session_state.user_input)

        current_user_message = {"input": st.session_state.user_input}


        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            result = st.session_state.agent.run_agent(input=st.session_state.user_input, callbacks=[st_callback])
            st.session_state.result = result
            response = result.get('output', '')
            st.session_state.messages.append({"roles": "assistant", "content": response})
            st.write(response)

            current_assistant_response = {"output": response}

        current_messages = [current_user_message, current_assistant_response] 


    with st.expander("View Document Sources"):
        if len(st.session_state.doc_sources) != 0:

            for document in st.session_state.doc_sources:
                st.divider()
                st.subheader("Source Content:")
                st.write(document.page_content)
                st.subheader("Metadata:")
                for key, value in document.metadata.items():
                    st.write(f"{key}: {value}")
        else:
                st.write("No document sources found")

    
    
    if st.session_state.summary is not None:
        with st.expander("Show Summary"):
            st.subheader("Summarization")
            result_summary = st.session_state.summary
            st.write(result_summary)


    buttons_placeholder = st.container()
    with buttons_placeholder:
        #st.button("Regenerate Response", key="regenerate", on_click=st.session_state.agent.regenerate_response)
        st.button("Clear Chat", key="clear", on_click=reset_chat)

    #st.write(st.session_state.history)
    #st.write(st.session_state.messages)
    #st.write(st.session_state.vector_store)
    st.write(st.session_state.br18_vectorstore)
    st.write(st.session_state.br18_appendix_child_vectorstore)
    #st.write(st.session_state.usc_vectorstore)
    st.write(st.session_state.agent)
    #st.write(st.session_state.result)


if __name__== '__main__':
    main()




    