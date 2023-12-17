import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes, clean_bullets, clean_trailing_punctuation, clean_ordered_bullets
from typing import List
import uuid
import json
import os
import re
from langchain.schema import Document


class CustomMarkdownHeaderTextSplitter(MarkdownHeaderTextSplitter):
    def split_text(self, text: str, original_metadata: dict) -> List[Document]:
        # Use the parent class's split_text method to split the text
        split_documents = super().split_text(text)

        # Add the original metadata to each split document
        for doc in split_documents:
            doc.metadata.update(original_metadata)

        return split_documents


class DocumentProcessor:

    @staticmethod
    def build_index_schema(documents):
        schema = {"text": []}

        if st.session_state.get('headers_to_split_on'):
            # Use headers from session state
            for _, header_name in st.session_state.headers_to_split_on:
                schema["text"].append({"name": header_name})
        else:
            # Create a dictionary to store headers and their corresponding levels
            header_levels = {}

            # Iterate through each document to find unique headers and determine their levels
            for doc in documents:
                for header, _ in doc.metadata.items():
                    level = header.count("#")  # Count number of '#' to determine the level
                    header_levels[header] = level

            # Sort headers based on their levels
            sorted_headers = sorted(header_levels.keys(), key=lambda h: header_levels[h])

            # Add sorted headers to the schema
            for header in sorted_headers:
                schema["text"].append({"name": header})

        # Convert schema to JSON and save to file
        with open(os.path.join("json/schema", "index_schema.json"), "w") as file:
            json.dump(schema, file, indent=4)

        st.success("Generated Indexing Schema Successfully")

        return documents
    
    def build_toc_from_documents(documents):
        toc = []

        headers_to_split_on = st.session_state.headers_to_split_on

        header_names = [header for _, header in headers_to_split_on]

        for doc in documents:
            header_data = {}
            for _, header in headers_to_split_on:
                if header in doc.metadata:
                    header_data[header] = re.sub(r"\s*-\s*", " ", doc.metadata[header])

            # Initialize a variable to keep track of the current path in the TOC structure
            current_path = []

            for header in header_names:
                if header in header_data and header_data[header]:
                    # Truncate current_path to the current level
                    current_path = current_path[:header_names.index(header)]
                    current_path.append((header, header_data[header]))

                    # Navigate and update the TOC structure based on the current_path
                    current_level = toc
                    for path_header, path_name in current_path[:-1]:
                        # Navigate to the correct level in the TOC
                        for entry in current_level:
                            if entry.get(path_header) == path_name:
                                current_level = entry.get('subsections', [])
                                break
                        else:
                            # Create a new entry if it doesn't exist at this level
                            new_entry = {path_header: path_name, 'subsections': []}
                            current_level.append(new_entry)
                            current_level = new_entry['subsections']

                    # Add the final element in the path
                    final_header, final_name = current_path[-1]
                    if not any(entry.get(final_header) == final_name for entry in current_level):
                        current_level.append({final_header: final_name, 'subsections': []})


        toc_file_path = os.path.join("json/toc", "generated_toc_2.json")
        with open(toc_file_path, 'w') as file:
            json.dump(toc, file, indent=4)

        st.success("Generated TOC Successfully")

        return documents

    @staticmethod
    def clean_chunks_content(documents):
        cleaning_functions = [
            clean_extra_whitespace, 
            clean_dashes,
            clean_bullets, 
            clean_trailing_punctuation, 
            clean_ordered_bullets
        ]

        for document in documents:
            cleaned_text = document.page_content
            for cleaning_function in cleaning_functions:
                # Apply each cleaning function to the text
                cleaned_text = cleaning_function(cleaned_text)
            
            # Update the document's page_content with the cleaned text
            document.page_content = cleaned_text

        # st.markdown("### Cleaned Document Chunks:")
        # st.write(documents)
        return documents

    
    @staticmethod
    def customize_document_metadata(documents):
        remove_metadata_keys = st.session_state.get('remove_metadata_keys',[])
        add_metadata_keys = st.session_state.get('add_metadata_keys', ['unique_id'])
        unique_id_type = st.session_state.get('unique_id_type', 'file_name + uuid')

        for chunk in documents:
            # Filter out selected metadata keys
            chunk.metadata = {k: v for k, v in chunk.metadata.items() if k not in remove_metadata_keys}

            # Add unique ID based on selected type if 'unique_id' is in added metadata
            if 'unique_id' in add_metadata_keys:
                if unique_id_type == 'uuid':
                    unique_id = str(uuid.uuid4())
                elif unique_id_type == 'file_name + uuid':
                    unique_id = f"{chunk.metadata.get('file_name', 'unknown')}_{uuid.uuid4()}"
                chunk.metadata['unique_id'] = unique_id

        # st.markdown("### Customized Document Metadata:")
        # st.write(documents)
        return documents
        
    @staticmethod
    def filter_short_documents(document_chunks):
        """
        Removes documents with page content of 50 characters or less.

        Args:
            document_chunks (List[Document]): A list of Document objects.

        Returns:
            List[Document]: A list of Document objects with the short ones removed.
        """
        filtered_documents = [doc for doc in document_chunks if len(doc.page_content) > 50]

        # st.markdown("### Filter Short Documents:")
        # st.write(filtered_documents)

        return filtered_documents
    
        