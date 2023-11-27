import os
import re
import uuid
import tempfile
import streamlit as st
import pypdf
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import time
from lxml import html
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from typing import Any, Optional
from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes, clean_bullets, clean_trailing_punctuation, clean_ordered_bullets
from unstructured.documents.elements import Table, Text


class DBStore:
    def __init__(self, client_db, file_name, collection_name=None):
        self.file_name = os.path.splitext(file_name)[0] if file_name else None
        st.session_state.document_filename = self.file_name

        self.file_path = None
        self.reader = None
        self.metadata = None

        self.embeddings = OpenAIEmbeddings()
        self.client = client_db.client
        self.collection_name = collection_name or client_db.collection_name

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
            doc_id = f"{self.file_name}_{uuid.uuid4()}"

            metadata = {
            "page_number": page_num,
            "file_name": self.file_name,
            "unique_id": doc_id,
            **self.metadata,
            }
        
        # Filter out any None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            doc = Document(
                page_content=page,
                metadata=metadata,
            )
            doc_chunks.append(doc)

        doc_chunks = filter_complex_metadata(doc_chunks)

        #st.write(doc_chunks)
        return doc_chunks
    
    def get_pdf_text(self):
        pages, metadata = self.parse_pdf()  # We only need the pages from the tuple
        cleaned_text_pdf = self.clean_text(pages)
        document_chunks = self.text_to_docs(cleaned_text_pdf)
        return document_chunks

    def get_vectorstore(self):
        document_chunks = self.get_pdf_text()

        ids = [doc.metadata["unique_id"] for doc in document_chunks]

        #st.write(f"Using collection name: {self.collection_name}")

        vectorstore = Chroma.from_documents(
            documents=document_chunks, 
            embedding=self.embeddings, 
            ids=ids,
            client=self.client, 
            collection_name=self.collection_name
        )

        return vectorstore
    
    def ingest_document(self, file, selected_collection_name):
        if not selected_collection_name or selected_collection_name == "None":
            raise ValueError("A valid collection must be selected for ingestion.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file.getvalue())
            self.file_path = tmpfile.name
            
            self.reader = pypdf.PdfReader(self.file_path)
            self.metadata = self.extract_metadata_from_pdf()

            vector_store = self.get_vectorstore()
            st.session_state.vector_store = vector_store

            self.collection_name = selected_collection_name

    def check_document_exists(self, collection_obj):
        """
        Check if a document with the same file name already exists in the Chroma database.
        
        Args:
        - collection: The Chroma collection object
        
        Returns:
        - bool: True if document exists, False otherwise
        """
        # Fetch documents from the Chroma collection with the same filename
        matching_documents = collection_obj.get(where={"file_name": {"$eq": self.file_name}}, include=["documents"])
        document_ids = matching_documents.get("ids", [])
        
        return document_ids


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
    

class DBIndexing:
    def __init__(self, client_db, file_name, collection_name=None):
        self.file_name = os.path.splitext(file_name)[0] if file_name else None
        st.session_state.document_filename = self.file_name

        self.file_path = None
        self.reader = None
        self.metadata = None

        self.embeddings = OpenAIEmbeddings()
        self.client = client_db.client
        self.collection_name = collection_name or client_db.collection_name

    def split_list(self, input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

    def partition_and_load(self, path_to_pdf):

        # Partition the PDF
        partitioned_elements = partition_pdf(
            filename=path_to_pdf,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy=st.session_state.get('chunking_strategy', 'by_title'),
            strategy=st.session_state.get('strategy', 'auto'),
            max_characters=st.session_state.get('max_characters', 4000),
            new_after_n_chars=st.session_state.get('new_after_n_chars', 3800),
            combine_text_under_n_chars=st.session_state.get('combine_text_under_n_chars', 2000),
            image_output_dir_path=os.path.dirname(path_to_pdf),
        )

        # Categorize the elements by type
        class Element(BaseModel):
            type: str
            text: Any
            metadata: Optional[dict] = None

        categorized_elements = []
        for raw_element in partitioned_elements:
        # Check if the element is a Table instance
            if isinstance(raw_element, Table):
                element_type = 'table'
            elif isinstance(raw_element, Text):
                element_type = 'text'
            else:
                element_type = 'unknown'  # Or handle other types as needed
            
            element_text = str(raw_element)
            element_metadata = raw_element.metadata.to_dict() if raw_element.metadata else None
            categorized_elements.append(Element(type=element_type, text=element_text, metadata=element_metadata))

        # Save the tables and text elements for later processing
        self.table_elements = [e for e in categorized_elements if e.type == "table"]
        self.text_elements = [e for e in categorized_elements if e.type == "text"]

        st.write(f"Number of table elements: {len(self.table_elements)}")
        st.write(f"Number of text elements: {len(self.text_elements)}")

        st.write(categorized_elements)

    def extract_metadata_from_pdf(self):
        """Extract metadata from the PDF file."""
        if self.reader and self.reader.metadata:
            metadata = self.reader.metadata
            essential_file_metadata = {
                "title": metadata.get("/Title", "").strip(),
                "author": metadata.get("/Author", "").strip(),
                "creation_date": metadata.get("/CreationDate", "").strip(),
            }
            st.session_state.document_metadata = essential_file_metadata
            return essential_file_metadata
        else:
            return {}

    def extract_metadata_from_element(self, element):
        """Extract metadata from an unstructured element."""

        essential_element_metadata = {
            "filetype": element.metadata.get('filetype'),
            "page_number": element.metadata.get('page_number'),
        }
        # Return the metadata dictionary
        #st.write(essential_element_metadata)

        return essential_element_metadata
    
    
    @staticmethod
    def merge_hyphenated_words(text):
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    @staticmethod
    def remove_dots(text):
        # Replace sequences of three or more dots with a single space.
        return re.sub(r'\.{4,}', ' ', text)

    def clean_text(self, elements):

        cleaning_functions = [
            clean_extra_whitespace,  # replaces fix_newlines and remove_multiple_newlines
            clean_dashes,            # new addition
            clean_bullets,           # new addition
            clean_trailing_punctuation, 
            clean_ordered_bullets, # new addition
            self.remove_dots,
        ]
        
        cleaned_text_elements = []
        for element in elements:
            if element.type == 'text':  # Only clean text elements
                cleaned_text = element.text
                for cleaning_function in cleaning_functions:
                    cleaned_text = cleaning_function(cleaned_text)
                element.text = cleaned_text  # Update the text in the element
            # Tables are preserved as is
            cleaned_text_elements.append(element)
        
        st.write(cleaned_text_elements)
        return cleaned_text_elements
    
    def text_to_docs(self, cleaned_elements):

        doc_chunks = []
        for element in cleaned_elements:
            doc_id = f"{self.file_name}_{uuid.uuid4()}"
            
            # Extract metadata from the unstructured element
            element_metadata = self.extract_metadata_from_element(element)
            file_metadata = self.metadata
            
            # Combine the global metadata with the element-specific metadata
            metadata = {
                "file_name": self.file_name,
                "unique_id": doc_id,
                "element_type": element.type,
                **file_metadata,  # Incorporate global metadata from PDF
                **element_metadata,  # Incorporate metadata from the specific element          
            }
            
            # Filter out any None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            doc = Document(
                page_content=element.text,
                metadata=metadata
            )
            doc_chunks.append(doc)

        doc_chunks = filter_complex_metadata(doc_chunks)
        return doc_chunks
    
    def get_pdf_text(self):


        # First, we'll use partition_and_load to split the PDF based on our strategies.
        self.partition_and_load(self.file_path)
        
        # Clean the texts of the elements
        cleaned_elements = self.clean_text(self.text_elements + self.table_elements)
        
        # Create document chunks
        document_chunks = self.text_to_docs(cleaned_elements)
        return document_chunks

    def get_vectorstore(self, batch_size=None, delay=None):
        document_chunks = self.get_pdf_text()
        st.write(document_chunks)

        if len(document_chunks) > 500 and batch_size is not None and delay is not None:
            # Batching ingestion
            for chunk in self.split_list(document_chunks, batch_size):
                chunk_ids = [doc.metadata["unique_id"] for doc in chunk]
                vectorstore = Chroma.from_documents(
                    documents=chunk,
                    embedding=self.embeddings,
                    ids=chunk_ids,
                    client=self.client,
                    collection_name=self.collection_name
                )
                time.sleep(delay)
    
        else:
            # All ingestion
            ids = [doc.metadata["unique_id"] for doc in document_chunks]
            vectorstore = Chroma.from_documents(
                documents=document_chunks, 
                embedding=self.embeddings, 
                ids=ids,
                client=self.client, 
                collection_name=self.collection_name
            )
        return vectorstore
    
    def ingest_document(self, file, selected_collection_name, batch_size, delay):
        st.write("Starting document ingestion...")
        if not selected_collection_name or selected_collection_name == "None":
            raise ValueError("A valid collection must be selected for ingestion.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file.getvalue())
            self.file_path = tmpfile.name
            self.reader = pypdf.PdfReader(self.file_path)
            self.metadata = self.extract_metadata_from_pdf()
            
            vector_store = self.get_vectorstore(batch_size, delay)
            st.session_state.vector_store = vector_store

            self.collection_name = selected_collection_name

    def check_document_exists(self, collection_obj):
        """
        Check if a document with the same file name already exists in the Chroma database.
        
        Args:
        - collection: The Chroma collection object
        
        Returns:
        - bool: True if document exists, False otherwise
        """
        # Fetch documents from the Chroma collection with the same filename
        matching_documents = collection_obj.get(where={"file_name": {"$eq": self.file_name}}, include=["documents"])
        document_ids = matching_documents.get("ids", [])
        
        return document_ids



class PDFTextExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.reader = pypdf.PdfReader(file_path)

    def extract_pages_from_pdf(self):
        pages = []
        for page in self.reader.pages:
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append(text)
        return pages

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
        return re.sub(r'\.{4,}', ' ', text)

    def clean_text(self, text):
        cleaning_functions = [
            self.merge_hyphenated_words,
            self.fix_newlines,
            self.remove_multiple_newlines,
            self.remove_dots,
        ]
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        return text

    def get_pdf_text(self):
        pages = self.extract_pages_from_pdf()
        cleaned_pages = [self.clean_text(page) for page in pages]
        document_chunks = [Document(page_content=page) for page in cleaned_pages]
        return document_chunks