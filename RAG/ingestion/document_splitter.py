import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter
from typing import List
from langchain.schema import Document
import json
import os

class CustomMarkdownHeaderTextSplitter(MarkdownHeaderTextSplitter):
    def split_text(self, text: str, original_metadata: dict) -> List[Document]:
        # Use the parent class's split_text method to split the text
        split_documents = super().split_text(text)

        # Add the original metadata to each split document
        for doc in split_documents:
            doc.metadata.update(original_metadata)

        return split_documents

class DocumentSplitter:
    @staticmethod
    def save_headers_to_split_on(headers):
        file_path = os.path.join("json/headers_info", "headers_to_split_on.json")
        with open(file_path, 'w') as file:
            json.dump(headers, file, indent=4)


    @staticmethod
    def markdown_header_splitter(documents):
        md_header_splitted_documents = []
        
        # Get headers to split on from session state
        headers_to_split_on = st.session_state.headers_to_split_on

        # Save headers to a JSON file
        DocumentSplitter.save_headers_to_split_on(headers_to_split_on)

        # Recursive splitting logic
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.get('chunk_size', 3000),
            chunk_overlap=st.session_state.get('chunk_overlap', 200),
            separators=st.session_state.get('selected_separators', None),
        )

        markdown_splitter = CustomMarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        for doc in documents:
            header_documents = markdown_splitter.split_text(doc.page_content, doc.metadata)
            section_documents = text_splitter.split_documents(header_documents)
            md_header_splitted_documents.extend(section_documents)

        st.markdown("### Header MD Splitted Chunks:")
        st.write(md_header_splitted_documents)
        return md_header_splitted_documents
    
    
    @staticmethod
    def recursive_splitter(documents):
        # Initialize a list to store the final chunks
        recursive_splitted_documents = []
        
        # Recursive splitting logic
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.get('chunk_size', 3000),
            chunk_overlap=st.session_state.get('chunk_overlap', 200),
            separators=st.session_state.get('selected_separators', None),
        )

        for doc in documents:
            section_documents = text_splitter.split_documents([doc])
            recursive_splitted_documents.extend(section_documents)

        # st.markdown("### Recursive Splitted Document Chunks:")
        # st.write(recursive_splitted_documents)
        return recursive_splitted_documents
    
    def character_splitter(documents):
        character_splitted_documents = []
        
        # Recursive splitting logic
        text_splitter = CharacterTextSplitter(
            chunk_size=st.session_state.get('chunk_size', 3000),
            chunk_overlap=st.session_state.get('chunk_overlap', 200),
        )

        for doc in documents:
            section_documents = text_splitter.split_documents([doc])
            character_splitted_documents.extend(section_documents)

        st.markdown("### Character Splitted Document Chunks:")
        st.write(character_splitted_documents)
        return character_splitted_documents
