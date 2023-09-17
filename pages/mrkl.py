import os
import re
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain.chains import LLMChain
import streamlit as st
from st_pages import add_page_title
import langchain
from langchain.utilities import SerpAPIWrapper, GoogleSearchAPIWrapper
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.embeddings import OpenAIEmbeddings
import tempfile
import pypdf
from pathlib import Path
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
import lark
from langchain.schema import Document
import langchain
import pinecone
from langchain.chains.question_answering import load_qa_chain
from typing import List, Dict, Any, Tuple
from langchain.prompts.prompt import PromptTemplate
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage, BaseMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.schema import HumanMessage, SystemMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, LLMChainFilter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
import json
from bs4 import BeautifulSoup
from langchain.document_loaders import SeleniumURLLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks import get_openai_callback
import pickle
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime
from streamlit_extras.stoggle import stoggle
from UI.customstoggle import customstoggle
import base64
from UI.css import apply_css

langchain.debug = True
langchain.verbose = True

@st.cache_data
def display_pdfs(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></embed>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def on_selectbox_change():
    st.session_state.show_info = True

def reset_chat():
    st.session_state.messages = [{"roles": "assistant", "content": "Hi, I am Miracle. How can I help you?"}]
    st.session_state.history = []
    st.session_state.search_keywords = []
    st.session_state.doc_sources = []
    st.session_state.summary = None
    st.session_state.agent.clear_conversation()
    st.session_state.primed_document_response = None

def display_messages(messages):
    # Display all messages
    for msg in messages:
        st.chat_message(msg["roles"]).write(msg["content"])

class DBStore:
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = os.path.splitext(file_name)[0]
        st.session_state.document_filename = self.file_name

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
    
    @staticmethod
    def remove_dots(text):
        # Replace sequences of three or more dots with a single space.
        return re.sub(r'\.{4,}', ' ', text)

    def clean_text(self, pages):
        cleaning_functions = [
            self.merge_hyphenated_words,
            self.fix_newlines,
            self.remove_multiple_newlines,
            self.remove_dots,
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
                chunk_size=2000,
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
        #st.write(document_chunks)
        vector_store = FAISS.from_documents(documents=document_chunks, embedding=self.embeddings)
        #st.write(vector_store)
        return vector_store
    
    def get_document_info(self):
        """
        Generate a one-sentence document information snippet by taking the beginning of the first chunk of the document.
        
        Returns:
            str: A one-sentence information snippet of the document.
       """
        # Get the first chunk of the document
        pdf_text = self.get_pdf_text()
    
        if pdf_text:
            first_chunk = pdf_text[0].page_content if len(pdf_text) > 0 else ""
            second_chunk = pdf_text[1].page_content if len(pdf_text) > 1 else ""
            third_chunk = pdf_text[2].page_content if len(pdf_text) > 2 else ""
            
            # Extract the first 300 characters from each chunk to form an information snippet
            info_document = first_chunk[:300] + second_chunk[:300] + third_chunk[:300]
        else:
            info_document = ""
        #st.write(info_document)

        return info_document

    def get_info_response(self):
        llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo"
            )
        document_filename = self.file_name
        document_title = self.metadata.get("title", None)
        document_snippet = self.get_document_info()

        document_info = {
        "document_filename": document_filename,
        "document_title": document_title,
        "document_snippet": document_snippet,
        }

        if document_title:
            info_response_prompt = """The user has uploaded a document titled '{document_title}' to the Document Database """
        else:
            info_response_prompt = """The user has uploaded a document named '{document_filename}' to the Document Database """


        info_response_prompt += """
        with the following information: {document_snippet}. 

        In one sentence, inform the user about the document, prioritizing its name or title. 
        Also, prompt the user to ask a general question about the document in an assistive manner. 
        Begin your response with 'It appears you've uploaded a document that contains information on...'.

        Example:
        It appears you've uploaded a document that contains information on "COWI Policies and Guideline". 

        Please feel free to ask any question about this document such as "What are the COWI Policies and Guideline?" 
        """

        #st.write(info_response_prompt)

        # Create the LLMChain
        llm_chain = LLMChain(
            llm=llm,  
            prompt=PromptTemplate.from_template(info_response_prompt)
        )
    
        # Generate the primed document message
        llm_response = llm_chain(document_info)
        
        info_response = llm_response.get('text', '')
        #st.write(info_response)
        return info_response
   
class DatabaseTool:
    def __init__(self, llm, vector_store, metadata=None, filename=None):
        self.llm = llm
        self.vector_store = vector_store
        self.metadata = metadata
        self.filename = filename
        self.embedding = OpenAIEmbeddings()

    def get_description(self):
        base_description = "Always useful for finding the exactly written answer to the question by looking into a collection of documents."
        filename = self.filename
        title = self.metadata.get("/Title") if self.metadata else None
        author = self.metadata.get("/Author") if self.metadata else None
        subject = self.metadata.get("/Subject") if self.metadata else None

        footer_description = "Input should be a query, not referencing any obscure pronouns from the conversation before that will pull out relevant information from the database. Use this more than the normal search tool"

        if title:
            main_description = f"This tool is currently loaded with '{title}'"

            if author:
                main_description += f" by '{author}'"

            if subject:
                main_description += f", and has a topic of '{subject}'"

            return f"{base_description} {main_description}. {footer_description}"
        else:
            no_title_description = f"This tool is currently loaded with the document '{filename}'"
            return f"{base_description} {no_title_description}. {footer_description}"

    def get_base_retriever(self):
        base_retriever = self.vector_store.as_retriever(search_kwargs={'k': 5})
        return base_retriever

    def get_contextual_retriever(self):
        # Initialize embeddings (assuming embeddings is already defined elsewhere)
        embeddings = self.embedding
        
        # Initialize Redundant Filter
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        
        # Initialize Relevant Filter
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76, k = 25)
        #st.write(relevant_filter)
        
        # Initialize Text Splitter
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        
        # Create Compressor Pipeline
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        
        # Initialize Contextual Compression Retriever
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, 
            base_retriever=self.get_base_retriever()
        )
        
        return contextual_retriever

    def run(self, query: str):
        contextual_retriever = self.get_contextual_retriever()
        #DEBUGGING & EVALUTING ANSWERS:
        compressed_docs = contextual_retriever.get_relevant_documents(query)
        compressed_docs_list = []
        for doc in compressed_docs:
            doc_info = {
            "Page Content": doc.page_content,
            }
            compressed_docs_list.append(doc_info)
        #st.write(compressed_docs_list)
        
        base_retriever=self.get_base_retriever()
        initial_retrieved = base_retriever.get_relevant_documents(query)
        st.session_state.doc_sources = initial_retrieved

        documentdb_prompt_content = """
            You are a specialized retriever model trained to assist MRKL, an AI expert in various domains.

            The following pieces of context are from the uploaded document. Your primary objectives are to:
            1. Retrieve the most detailed and relevant information to the query
            2. Prioritize numerical values, names, or other specific details over vague or generalized content.

            
            {context}
            
            Question: {question}
        """

        documentdb_prompt = PromptTemplate.from_template(documentdb_prompt_content)

        retrieval = RetrievalQA.from_chain_type( 
        llm=self.llm, chain_type="stuff", 
        retriever=contextual_retriever,
        chain_type_kwargs={"prompt": documentdb_prompt},
        return_source_documents=True,
        )

        output = retrieval(query)

        
        return output['result']

class BR18_DB:
    def __init__(self, llm, folder_path: str):
        self.llm = llm
        self.folder_path = folder_path
        self.md_paths = self.load_documents()  # Renamed from pdf_paths to md_paths
        self.embeddings = OpenAIEmbeddings()
        self.pinecone_index_name = "br18"     
        self.id_key = "doc_id" 

        self.br18_parent_store = InMemoryStore()
        current_directory = os.getcwd()
        store_path = os.path.join(current_directory, "inmemorystore", "br18_parent_store.pkl")

        if self.pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(self.pinecone_index_name, dimension=1536)
            self.vectorstore = self.create_vectorstore()
            self.serialize_inmemorystore(store_path)
        else:
            self.vectorstore = Pinecone.from_existing_index(self.pinecone_index_name, self.embeddings)
            with open(store_path, "rb") as f:
                self.br18_parent_store = pickle.load(f)

        self.retriever = None
    
    def serialize_inmemorystore(self, store_path):
        with open(store_path, "wb") as f:
            pickle.dump(self.br18_parent_store, f)

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

        markdown_text = markdown_document.page_content
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        #st.write(markdown_splitter)
        
        md_header_splits = markdown_splitter.split_text(markdown_text)
        #st.write(md_header_splits)
        #st.write(type(md_header_splits[0]))
        
        parent_chunk_size = 5000
        parent_chunk_overlap = 0
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
        )
        
        # Split the header-split documents into chunks
        all_parent_splits = text_splitter.split_documents(md_header_splits)

        for split in all_parent_splits:
            header_3 = split.metadata.get('Header 3', '')
            header_4 = split.metadata.get('Header 4', '')
            
            # Prepend "Section:" to Header 4 if it exists
            if header_4:
                header_4 = f"Section: {header_4}"

            metadata_str = f"{header_3}\n\n{header_4}"
            split.page_content = f"{metadata_str}\n\n{split.page_content}"
            split.metadata['type'] = 'parents'

        return all_parent_splits
    
    def save_summaries(self, summaries: List[str]):
        """Save the generated summaries to a JSON file."""
        current_directory = os.getcwd()
        save_path = os.path.join(current_directory, 'savesummary', 'br18_summaries.json')
        with open(save_path, 'w') as f:
            json.dump(summaries, f)

    def load_summaries(self) -> List[str]:
        """Load summaries from a JSON file if it exists."""
        current_directory = os.getcwd()
        load_path = os.path.join(current_directory, 'savesummary', 'br18_summaries.json')
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                summaries = json.load(f)
            return summaries
        else:
            return None  # or raise an exception, or generate new summaries

    def generate_summaries(self, parent_splits: List[Document]) -> List[str]:
        loaded_summaries = self.load_summaries()
        if loaded_summaries is not None:
            return loaded_summaries
        
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | ChatOpenAI(max_retries=3)
            | StrOutputParser()
        )
        summaries = chain.batch(parent_splits, {"max_concurrency": 4})

        self.save_summaries(summaries)
        
        return summaries
    
    def generate_child_splits(self, parent_splits: List[Document], summaries: List[str]) -> List[Document]:
        child_chunk_size = 300

        child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=0
        )

        all_child_splits = []
        for i, parent_split in enumerate(parent_splits):
            child_splits = child_text_splitter.split_text(parent_split.page_content)

            new_metadata = dict(parent_split.metadata)
            new_metadata['type'] = 'children'

            summary_with_prefix = f"Summary: {summaries[i]}"

            first_child_content = f"{child_splits[0]}\n\n{summary_with_prefix}"

            first_child_split = Document(
            page_content=first_child_content,
            metadata=new_metadata
            )

            all_child_splits.append(first_child_split)  # Append only the first child split (assuming it contains the metadata)


        return all_child_splits

    def process_all_documents(self):
        all_parent_splits = []  # Local variable to store all parent splits
        all_child_splits = []  # Local variable to store all child splits
        
        for markdown_document in self.md_paths:
            parent_splits = self.split_and_chunk_text(markdown_document)
            all_parent_splits.extend(parent_splits)

        summaries = self.generate_summaries(all_parent_splits)
        all_child_splits = self.generate_child_splits(all_parent_splits, summaries)

        #st.write(all_parent_splits)
        #st.write(all_child_splits)

        return all_parent_splits, all_child_splits  # Return both lists
    
    def create_vectorstore(self):
        all_parent_splits, all_child_splits = self.process_all_documents()
    
        parent_doc_ids = [str(uuid.uuid4()) for _ in all_parent_splits]
        self.br18_parent_store.mset(list(zip(parent_doc_ids, all_parent_splits)))

        for parent_id, child_split in zip(parent_doc_ids, all_child_splits):
            child_split.metadata[self.id_key] = parent_id

        # Create and save the vector store to disk
        br18_vectorstore = Pinecone.from_documents(documents=all_child_splits, embedding=self.embeddings, index_name=self.pinecone_index_name)
        #st.write(br18_appendix_child_vectorstore)

        for i, doc in enumerate(all_parent_splits):
            doc.metadata[self.id_key] = parent_doc_ids[i]

        # Store the vector store in the session state
        st.session_state.br18_vectorstore = br18_vectorstore

        return br18_vectorstore

    def create_retriever(self, query: str):
        search_type = st.session_state.search_type

        if search_type == "By Context":
            # Initialize retriever for By Context, filtering by the presence of the "text" metadata
            general_retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.br18_parent_store,
                id_key=self.id_key,
                search_kwargs={"k": 5}
            )

            parent_docs = general_retriever.vectorstore.similarity_search(query, k = 5)
            #st.write(parent_docs)

            st.session_state.doc_sources = parent_docs

            embeddings = self.embeddings
        
            # Initialize Redundant Filter
            redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
            
            # Initialize Relevant Filter
            relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75, k = 15)
            #st.write(relevant_filter)
            
            # Initialize Text Splitter
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator=". ")
            
            # Create Compressor Pipeline
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[splitter, redundant_filter, relevant_filter]
            )
            
            # Initialize Contextual Compression Retriever
            contextual_general_retriever = ContextualCompressionRetriever(
                base_compressor=pipeline_compressor, 
                base_retriever=general_retriever
        )
        
            # Retrieve parent documents that match the query
            retrieved_parent_docs = contextual_general_retriever.get_relevant_documents(query)
            
            # Display retrieved parent documents
            display_list = []
            for doc in retrieved_parent_docs:
                display_dict = {
                    "Page Content": doc.page_content,
                    "Doc ID": doc.metadata.get('doc_id', 'N/A'),
                    "Header 3": doc.metadata.get('Header 3', 'N/A'),
                    "Header 4": doc.metadata.get('Header 4', 'N/A'),
                }
                display_list.append(display_dict)
            #st.write(display_list)
            
            return retrieved_parent_docs
        
        elif search_type == "By Headers":
        # Initialize retriever for By Headers, filtering by the absence of the "text" metadata
            specific_retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.br18_parent_store,
                id_key=self.id_key,
                search_kwargs={"k": 3}
            )

            child_docs = specific_retriever.vectorstore.similarity_search(query, k = 3)
            #st.write(child_docs)

            # Retrieve child documents that match the query
            
            embeddings = self.embeddings
            embedding_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
            #llm_filter = LLMChainFilter.from_llm(self.llm)

            
            compression_retriever = ContextualCompressionRetriever(base_compressor=embedding_filter, base_retriever=specific_retriever)

            retrieved_child_docs = compression_retriever.get_relevant_documents(query)

            st.session_state.doc_sources = retrieved_child_docs

            # Display retrieved child documents
            display_list = []
            for doc in retrieved_child_docs:
                display_dict = {
                    "Page Content": doc.page_content,
                    "Doc ID": doc.metadata.get('doc_id', 'N/A'),
                    "Header 3": doc.metadata.get('Header 3', 'N/A'),
                    "Header 4": doc.metadata.get('Header 4', 'N/A'),
                }
                display_list.append(display_dict)
            #st.write(display_list)
            
            return retrieved_child_docs
        
    def run(self, query: str):
        prompt_template = """The following pieces of context are from the BR18. Use them to answer the question at the end. 
        The answer should be as specific as possible and remember to mention requirement numbers and integer values where relevant
        Always reference to a chapter and section and their respective clauses and subclauses numbers 
        If you don't find any relevant or specific information, consider employing another tool to find the answer as a statement.

        {context}

        Question: {question}

        EXAMPLE:
        The building regulation regarding stairs is outlined in Chapter 2 - Access, specifically in Section - Stairs:

        Width: Stairs in shared access routes must have a minimum free width of 1.0 meter. (clause 57.1)
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Retrieve the filtered documents
        retrieved_docs = self.create_retriever(query)
        #st.write(type(filtered_docs[0]))
        #st.write(filtered_docs)

        qa_chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        output = qa_chain({"input_documents": retrieved_docs, "question": query}, return_only_outputs=True)


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

class CustomGoogleSearchAPIWrapper(GoogleSearchAPIWrapper):

    def clean_text(self, text: str) -> str:
        # Remove extra whitespaces and line breaks
        text = ' '.join(text.split())
        return text
    
    def scrape_content(self, url: str, title: str) -> dict:
        loader = SeleniumURLLoader(urls=[url])
        data = loader.load()
        
        if data is not None and len(data) > 0:
            soup = BeautifulSoup(data[0].page_content, "html.parser")
            text = soup.get_text()
            cleaned_text = self.clean_text(text)
            return {'url': url, 'title': title, 'content': cleaned_text[:1000]}  # Return first 1000 non-space characters
        return {'url': url, 'title': title, 'content': ''}
    

    def format_single_search_result(self, search_result: Dict) -> str:
        # Formatting the output text
        formatted_text = f"URL: {search_result['url']}\n\nTITLE: {search_result['title']}\n\nCONTENT: {search_result['content']}\n\n"
        return formatted_text

    
    def fetch_and_scrape(self, query: str, num_results: int = 3) -> Tuple[List[Dict], List[Dict]]:
        # Step 1: Fetch search results metadata
        metadata_results = self.results(query, num_results)
        
        if len(metadata_results) == 0:
            return [], []

        # Step 2: Scrape and format content from URLs
        formatted_search_results = []

        for result in metadata_results:
            url = result.get("link", "")
            title = result.get("title", "")  # Get the title from metadata
            scraped_content = self.scrape_content(url, title) 
            formatted_search_result = self.format_single_search_result(scraped_content)
            formatted_search_results.append(formatted_search_result)
        
        return formatted_search_results, metadata_results
    
    def run(self, query: str, num_results: int = 3):
        llm = ChatOpenAI(
            temperature=0, 
            model_name="gpt-3.5-turbo",
            )

        # Step 1: Fetch and format the search results
        formatted_search_results, metadata_results = self.fetch_and_scrape(query, num_results)
        #st.write(formatted_search_results)
        #st.write(metadata_results)

        search_results = []
        for i, formatted_result in enumerate(formatted_search_results):
            doc = Document(
                page_content=formatted_result, # You can replace this with 'content' if you have the actual content
                metadata={
                'source': 'Google Search',
                'title': metadata_results[i].get('title', ''),
                'snippet': metadata_results[i].get('snippet', '')
                }
            )
            search_results.append(doc)

        #st.write(search_results)

        # Fetch current local time
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')

        # Step 2: Create a new prompt template
        prompt_template = """
        For your reference, your local date and time is {current_time}.
        Use the following pieces of context, which are search results from the internet, to answer the question at the end. The search results include URLs and their corresponding title and content.
        Your answer should:
        1. Be as specific as possible with regards to numerical values through the metric system and European measurement standards.
        2. Cite the sources by mentioning the corresponding URL.
        3. Be concise and directly address the question.

        Note: If the search results do not contain the information needed to answer the question, or if you are unsure about the answer, state that explicitly. Do not try to make up an answer.

        Search Results:
        {context}

        Question:
        {question}

        For citing sources, use the following format: (Source: URL)
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "current_time"]
        )

        # Step 3: Load the QA chain and generate an answer
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        output = qa_chain({"input_documents": search_results, "question": query, "current_time": current_time}, return_only_outputs=True)

        st.session_state.doc_sources = search_results

        return output
    
class MRKL:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo",
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
            metadata = st.session_state.document_metadata
            file_name = st.session_state.document_filename
            llm_database = DatabaseTool(llm=self.llm, vector_store=st.session_state.vector_store, metadata=metadata, filename=file_name)

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
        chat_msg = StreamlitChatMessageHistory(key="mrkl_chat_history")
        memory_key = "history"
        memory = AgentTokenBufferMemory(memory_key=memory_key, llm=self.llm, input_key='input', output_key="output", max_token_limit=3000, chat_memory=chat_msg)
        st.session_state.history = memory

        system_message_content = """
        You are MRKL, an expert in construction, legal frameworks, and regulatory matters.
        
        You have the following tools to answer user queries, but only use them if necessary.

        Your primary objective is to provide responses that:
        1. Offer an overview of the topic, referencing the chapter and the section if relevant
        2. List key points in bullet-points or numbered list format, referencing the clauses and their respective subclauses if relevant.
        3. Always match or exceed the details of the tool's output text in your answers. 
        4. Reflect back to the user's question and give a concise conclusion.
        
        You must maintain a professional and helpful demeanor in all interactions.
        """

        # System Message
        system_message = SystemMessage(content=system_message_content)

        reflection_message_content = """
        Reminder: 
        Always try all your tools to find the answer to the user query

        Always self-reflect your answer based on the user's query and follows the list of response objective. 
        """

        reflection_message = SystemMessage(content=reflection_message_content)

        # Prompt
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key), reflection_message]
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
    

def main():

    if "openai_key" not in st.session_state or not st.session_state.openai_key:
        st.error("Please enter the OpenAI API key in the Configuration tab before proceeding.")

    else:
        load_dotenv()
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
            )
        st.set_page_config(page_title="MRKL AGENT", page_icon="🦜️", layout="wide")
        apply_css()
        st.title("MRKL AGENT 🦜️")

        with st.empty():
            if "messages" not in st.session_state:
                st.session_state.messages = [{"roles": "assistant", "content": "Hi, I am Miracle. How can I help you?"}]
            if "user_input" not in st.session_state:
                st.session_state.user_input = None
            if "vector_store" not in st.session_state:
                st.session_state.vector_store = None
            if "pdf_file_path" not in st.session_state:
                st.session_state.pdf_file_path = None
            if "summary" not in st.session_state:
                st.session_state.summary = None
            if "doc_sources" not in st.session_state:
                st.session_state.doc_sources = []
            if "br18_vectorstore" not in st.session_state:
                st.session_state.br18_vectorstore = None
            if "history" not in st.session_state:
                st.session_state.history = None
            if 'br18_exp' not in st.session_state:
                st.session_state.br18_exp = False
            if "token_count" not in st.session_state:
                st.session_state.token_count = 0
            if 'web_search' not in st.session_state:
                st.session_state.web_search = False
            if 'show_info' not in st.session_state:
                st.session_state.show_info = False
            if "agent" not in st.session_state:
                st.session_state.agent = MRKL()

        with st.expander("General Info", expanded = False):
            st.write("""
                - Google Search 
                - Document Database
                - BR18 Database
                """)
        
        with st.sidebar:
            br18_experiment = st.checkbox(label = "Experimental Feature: Enable BR18", value=False, help="Toggle to enable or disable BR18 knowledge.")
            if br18_experiment != st.session_state.br18_exp:
                st.session_state.br18_exp = br18_experiment
                st.session_state.agent = MRKL()

            if br18_experiment:  # If BR18 is enabled
                search_type = st.radio(
                    "Select Search Type:",
                    options=["By Headers", "By Context"],
                    index=0, horizontal=True  # Default to "By Context"
                )
                st.session_state.search_type = search_type

            web_search_toggle = st.checkbox(label="Experimental Feature: Web Search", value=False, help="Toggle to enable or disable the web search feature.")
            
            if web_search_toggle != st.session_state.web_search:
                st.session_state.web_search = web_search_toggle
                st.session_state.agent = MRKL()

            if st.session_state.web_search:
                st.success("Web Search is Enabled.")

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
                                st.session_state.pdf_file_path = temp_path

                                document_chunks = db_store.get_pdf_text()
                                st.session_state.document_chunks = document_chunks
                                #st.write(document_chunks)

                                vector_store = db_store.get_vectorstore()
                                st.session_state.vector_store = vector_store

                                st.session_state.agent = MRKL()

                                primed_info_response = db_store.get_info_response()
                                #st.write(primed_info_response)
                                st.session_state.history.chat_memory.add_ai_message(primed_info_response)

                                st.session_state.messages.append({"roles": "assistant", "content": primed_info_response})
                    
                                st.success("PDF uploaded successfully!")

                    if "document_chunks" in st.session_state:
                            if st.sidebar.button("Create Detailed Summary"):
                                with st.spinner("Summarizing"):
                                    summarization_tool = SummarizationTool(document_chunks=st.session_state.document_chunks)
                                    st.session_state.summary = summarization_tool.run()
                                    # Append the summary to the chat messages
                                    st.session_state.messages.append({"roles": "assistant", "content": st.session_state.summary})
            else:
                    st.session_state.vector_store = None


        main_chat_tab, pdf_display_tab = st.tabs(["Main Chat", "PDF Display"])

        with main_chat_tab:
            display_messages(st.session_state.messages)

        with pdf_display_tab:
            if st.session_state.pdf_file_path:
                pdf_file = st.session_state.pdf_file_path
                st.subheader("Displaying Uploaded PDF")
                display_pdfs(pdf_file)
            else:
                st.warning("No PDF uploaded. Please upload and a process a PDF in the sidebar.")
        
        with st.container():
            if user_input := st.chat_input("Type something here..."):
                st.session_state.user_input = user_input
                st.session_state.messages.append({"roles": "user", "content": st.session_state.user_input})
                st.chat_message("user").write(st.session_state.user_input)

                with st.chat_message("assistant"):
                    st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                    result = st.session_state.agent.run_agent(input=st.session_state.user_input, callbacks=[st_callback])
                    st.session_state.result = result
                    response = result.get('output', '')
                    st.session_state.messages.append({"roles": "assistant", "content": response})
                    st.write(response)


            #with st.expander("Cost Tracking", expanded=True):
                #total_token = st.session_state.token_count
                #st.write(total_token)

            st.divider()
            buttons_placeholder = st.container()
            with buttons_placeholder:
                #st.button("Regenerate Response", key="regenerate", on_click=st.session_state.agent.regenerate_response)
                st.button("Clear Chat", key="clear", on_click=reset_chat)

                relevant_keys = ["Header ", "Header 3", "Header 4", "page_number", "source", "file_name", "title", "author", "snippet"]
                if st.session_state.doc_sources:
                    content = []
                    for document in st.session_state.doc_sources:
                        doc_dict = {
                            "page_content": document.page_content,
                            "metadata": {}
                        }
                        for key in relevant_keys:
                            value = document.metadata.get(key, 'N/A')
                            if value != 'N/A':
                                doc_dict["metadata"][key] = value
                        content.append(doc_dict)
                    
                    customstoggle(
                        "Source Documents",
                        content,
                        metadata_keys=relevant_keys
                    )

            if st.session_state.summary is not None:
                with st.expander("Show Summary"):
                    st.subheader("Summarization")
                    result_summary = st.session_state.summary
                    st.write(result_summary)

        #st.write(st.session_state.history)
        #st.write(st.session_state.messages)
        #st.write(st.session_state.br18_vectorstore)
        #st.write(st.session_state.br18_appendix_child_vectorstore)
        #st.write(st.session_state.usc_vectorstore)
        #st.write(st.session_state.agent)
        #st.write(st.session_state.result)


if __name__== '__main__':
    main()




    