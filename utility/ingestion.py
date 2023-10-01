import os
import re
import uuid
import tempfile
import streamlit as st
import pypdf
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma






def ingest_documents_db(file, file_name, collection_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(file.getvalue())
        temp_path = tmpfile.name
        db_store = DBStore(temp_path, file_name, collection_name)

        st.session_state.pdf_file_path = temp_path
        st.session_state.db_store = db_store

        document_chunks = db_store.get_pdf_text()
        st.session_state.document_chunks = document_chunks

        vector_store = db_store.get_vectorstore()
        st.session_state.vector_store = vector_store



class DBStore:
    def __init__(self, file_path, file_name, collection_name):
        self.file_path = file_path
        self.file_name = os.path.splitext(file_name)[0]
        st.session_state.document_filename = self.file_name

        self.reader = pypdf.PdfReader(file_path)
        self.metadata = self.extract_metadata_from_pdf()
        self.embeddings = OpenAIEmbeddings()


        self.client = chromadb.HttpClient(settings=Settings(allow_reset=True))
        self.collection_name = collection_name

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

                doc_id = f"{self.file_name}_{uuid.uuid4()}"

                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page_number": page_num,
                        "chunk": i,
                        "source": f"p{page_num}-{i}",
                        "file_name": self.file_name,
                        "unique_id": doc_id,
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

        ids = [doc.metadata["unique_id"] for doc in document_chunks]

        vectorstore = Chroma.from_documents(
            documents=document_chunks, 
            embedding=self.embeddings, 
            ids=ids,
            client=self.client, 
            collection_name=self.collection_name
        )
        
        return vectorstore
    
    def reset_client(self):
        self.client.reset()