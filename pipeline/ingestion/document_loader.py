import streamlit as st
import tempfile
import os
import pypdf
from langchain.document_loaders import TextLoader
from unstructured.partition.md import partition_md

class DocumentLoader:
    
    @staticmethod
    def document_loader_langchain(uploaded_file):
        documents = []
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            file_path = tmpfile.name
        
        try:
            file_name = os.path.splitext(uploaded_file.name)[0]
            file_type = "markdown" if file_path.endswith('.md') else "pdf"

            # Check file type and process accordingly
            if file_path.endswith('.md'):
                loader = TextLoader(file_path)
                documents = loader.load()

            elif file_path.endswith('.pdf'):
                # If PDF, use a PDF loader
                with open(file_path, "rb") as f:
                    pdf_reader = pypdf.PdfReader(f)
                    # Load and process PDF file (not fully implemented)
                    documents = [page.extract_text() for page in pdf_reader.pages]
            else:
                raise ValueError("Unsupported file format. Please upload a .md or .pdf file.")
            
            for doc in documents:
                doc.metadata.update({"file_name": file_name, "file_type": file_type})
        
        finally:
            # Ensure that the temporary file is cleaned up
            os.remove(file_path)

        st.markdown("### Loaded Documents:")
        st.write(documents)
        return documents
    
    @staticmethod
    def document_loader_unstructured(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            file_path = tmpfile.name

        uploaded_document = partition_md(filename=file_path)

        os.remove(file_path)

        st.write(uploaded_document)
        return uploaded_document