import os
import re
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, Tool, AgentExecutor, ConversationalAgent, ZeroShotAgent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain.chains import LLMChain, StuffDocumentsChain
import streamlit as st
import langchain
from langchain.utilities import SerpAPIWrapper, GoogleSearchAPIWrapper
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
import tempfile
import pypdf
import openai
from pathlib import Path
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, LLMChainFilter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
import json
from bs4 import BeautifulSoup
from langchain.document_loaders import SeleniumURLLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks import get_openai_callback


langchain.debug = True
langchain.verbose = True

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
        st.write(compressed_docs_list)
        
        base_retriever=self.get_base_retriever()
        initial_retrieved = base_retriever.get_relevant_documents(query)

        retrieval = RetrievalQA.from_chain_type( 
        llm=self.llm, chain_type="stuff", 
        retriever=contextual_retriever,
        return_source_documents=True,
        )

        output = retrieval(query)
        st.session_state.doc_sources = initial_retrieved

        
        return output['result']

class BR18_DB:
    def __init__(self, llm, folder_path: str):
        self.llm = llm
        self.folder_path = folder_path
        self.md_paths = self.load_documents()  # Renamed from pdf_paths to md_paths
        self.embeddings = OpenAIEmbeddings()
        self.pinecone_index_name = "br18"
        self.br18_parent_store = InMemoryStore()
        self.id_key = "doc_id" 
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
        parent_chunk_size = 5000
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
        all_processed_splits = []
        for markdown_document in self.md_paths:
            processed_splits = self.split_and_chunk_text(markdown_document)
            all_processed_splits.extend(processed_splits)
        #st.write(all_processed_splits)
        return all_processed_splits
    
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
        search_type = st.session_state.search_type

        if search_type == "General Search":
            base_retriever = self.vectorstore.as_retriever(search_kwargs={'k': 5})
            compression_filter = LLMChainFilter.from_llm(self.llm)
            base_relevant_docs = base_retriever.get_relevant_documents(query)
            st.write(base_relevant_docs)

            compression_retriever = ContextualCompressionRetriever(
            base_compressor=compression_filter, 
            base_retriever=base_retriever
        )

            compressed_relevant_docs = compression_retriever.get_relevant_documents(query)
            st.write(compressed_relevant_docs)
            #st.write(type(relevant_docs))

            keywords = self.get_keywords(query)
            #st.write(keywords)

            filtered_docs = self.post_filter_documents(keywords, compressed_relevant_docs)
            st.write(filtered_docs)
            #st.write(type(filtered_docs[0]))

            if not filtered_docs:
                non_filtered = compressed_relevant_docs[:5]
                st.write(non_filtered)
                return non_filtered

            return filtered_docs
        
        if search_type == "Specific Search":
            specific_retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,  # Replace with the child vector store if needed
            docstore=self.br18_parent_store,  # Use the in-memory parent store
            id_key=self.id_key  # The key used to match parent and child documents
        )
            st.write(specific_retriever)
            
            retrieved_child_docs = specific_retriever.vectorstore.similarity_search(query)
            st.write(retrieved_child_docs)

            retrieved_parent_docs = specific_retriever.get_relevant_documents(query)[:3]
            st.write(retrieved_parent_docs)
            return retrieved_parent_docs

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
        retrieved_docs = self.create_retriever(query)
        #st.write(type(filtered_docs[0]))
        #st.write(filtered_docs)

        qa_chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        output = qa_chain({"input_documents": retrieved_docs, "question": query}, return_only_outputs=True)

        st.session_state.doc_sources = retrieved_docs

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

    def scrape_content(self, url: str) -> str:
        loader = SeleniumURLLoader(urls=[url])
        data = loader.load()
        
        if data is not None and len(data) > 0:
            soup = BeautifulSoup(data[0].page_content, "html.parser")
            text = soup.get_text()
            return text[:1000]  # Return first 1000 non-space characters
        return ''
    
    def fetch_and_scrape(self, query: str, num_results: int = 3) -> str:
        # Step 1: Fetch search results metadata
        metadata_results = self.results(query, num_results)
        if len(metadata_results) == 0:
            return "No good Google Search Result was found"
        
        # Step 2: Extract URLs
        urls = [result.get("link", "") for result in metadata_results if "link" in result]

        # Step 3: Scrape content from URLs
        texts = []
        for url in urls:
            scraped_content = self.scrape_content(url)
            texts.append(scraped_content)
        
        return " ".join(texts)[:3000]   # Return first 2000 characters combined from all URLs

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
        llm_search = CustomGoogleSearchAPIWrapper()

        current_directory = os.getcwd()

        tools = [
            Tool(
                name="Google_Search",
                func=llm_search.fetch_and_scrape,
                description="Useful when you cannot find a clear answer after looking up the database and that you need to search the internet for information. Input should be a fully formed question based on the context of what you couldn't find and not referencing any obscure pronouns from the conversation before"
            ),
            Tool(
                name='Calculator',
                func=llm_math.run,
                description='Useful for when you need to answer questions about math.'
            ),
        ]

        if st.session_state.vector_store is not None:
            metadata = st.session_state.document_metadata
            file_name = st.session_state.document_filename
            llm_database = DatabaseTool(llm=llm, vector_store=st.session_state.vector_store, metadata=metadata, filename=file_name)

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

            tools.extend([
            Tool(
                name='BR18_Database',
                func=llm_br18.run,
                description="""
                Always useful for when you need to answer questions about the Danish Building Regulation 18 (BR18). 
                Input should be the specific keywords from the user query. Exclude the following common terms and their variations or synonyms especially words such as "building" and "regulation".
                Use this tool more often than the normal search tool.
                """
            ),
            ])
        return tools

    def load_agent(self):
        llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo",
            )
        
        # Memory
        chat_msg = StreamlitChatMessageHistory(key="mrkl_chat_history")
        memory_key = "history"
        memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, input_key='input', output_key="output", max_token_limit=8000, chat_memory=chat_msg)
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
        Reminder: 
        Always try all your tools to find the right answer before saying 'I don't know,' with the search tool as your last resort.
        Always self-reflect your answer based on the user's query and follows the list of response objective. 
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
        with get_openai_callback() as cb:
            result = self.agent_executor({"input": input}, callbacks=callbacks)
            st.session_state.token_count = cb
            print(cb)
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
        st.session_state.history = None
    if 'br18_exp' not in st.session_state:
        st.session_state.br18_exp = False
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0

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

        if br18_experiment:  # If BR18 is enabled
            search_type = st.radio(
                "Select Search Type:",
                options=["General Search", "Specific Search"],
                index=0, horizontal=True  # Default to "General Search"
            )
            st.session_state.search_type = search_type

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



    display_messages(st.session_state.messages)


    if user_input := st.chat_input("Type something here..."):
        st.session_state.user_input = user_input
        st.session_state.messages.append({"roles": "user", "content": st.session_state.user_input})
        st.chat_message("user").write(st.session_state.user_input)

        current_user_message = {"input": st.session_state.user_input}


        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
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

    #with st.expander("Cost Tracking", expanded=True):
        #total_token = st.session_state.token_count
        #st.write(total_token)

    buttons_placeholder = st.container()
    with buttons_placeholder:
        #st.button("Regenerate Response", key="regenerate", on_click=st.session_state.agent.regenerate_response)
        st.button("Clear Chat", key="clear", on_click=reset_chat)

    


    #st.write(st.session_state.history)
    #st.write(st.session_state.messages)
    #st.write(st.session_state.vector_store)
    st.write(st.session_state.br18_vectorstore)
    #st.write(st.session_state.br18_appendix_child_vectorstore)
    #st.write(st.session_state.usc_vectorstore)
    st.write(st.session_state.agent)
    #st.write(st.session_state.result)


if __name__== '__main__':
    main()




    