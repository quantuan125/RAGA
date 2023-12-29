import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Iterable
import uuid
import os 

class DocumentIndexer:

    @staticmethod
    def save_docs_to_jsonl(array:Iterable[Document], file_name:str, file_path: str = None)->None:
        if file_path is None:
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the parent directory
            parent_dir = os.path.dirname(current_dir)
            # Construct the full file path to the inmemorystore folder
            file_path = os.path.join(parent_dir, "inmemorystore/indexed_documents", file_name)
        else:
            # Append the file name to the provided file_path
            file_path = os.path.join(file_path, file_name)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as jsonl_file:
            for doc in array:
                jsonl_file.write(doc.json() + '\n')

    @staticmethod
    def summary_indexing(documents):
        
        for doc in documents:
            # Check if unique_id exists, else create one
            unique_id = doc.metadata.get("unique_id", str(uuid.uuid4()))
            doc.metadata["unique_id"] = unique_id
        
        file_name = st.session_state.original_document_file_name

        # Save the original documents with unique_id
        DocumentIndexer.save_docs_to_jsonl(documents, file_name)

        summarize_prompt_template = """You are an assistant tasked with summarizing tables and text. \ 
        Give a concise summary of the table or text below. 
        Begin your answer with "The <text/table> outlines..."
        
        '''
        {documents} 
        '''
        """

        summarize_prompt = ChatPromptTemplate.from_template(summarize_prompt_template)

        # Summary chain
        model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

        summarize_chain = {"documents": lambda x: x.page_content} | summarize_prompt | model | StrOutputParser()

        summaries = summarize_chain.batch(documents, {"max_concurrency": 5})

        summary_docs = []

        for doc, summary in zip(documents, summaries):
            doc.page_content = summary
            summary_docs.append(doc)

        #DocumentIndexer.save_docs_to_jsonl(summary_docs, "summary_documents.jsonl")

        st.markdown("### Summarized Document Chunks:")
        st.write(summary_docs)
        
        return summary_docs
    
    @staticmethod
    def parent_document_indexing(documents):

        for doc in documents:
            unique_id = doc.metadata.get("unique_id", str(uuid.uuid4()))
            doc.metadata["unique_id"] = unique_id

        parent_file_name = st.session_state.original_document_file_name

        DocumentIndexer.save_docs_to_jsonl(documents, parent_file_name)

        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        child_docs = []

        for doc in documents:
            # Check if unique_id exists, else create one
            unique_id = doc.metadata.get("unique_id", str(uuid.uuid4()))

            # Split into smaller chunks
            sub_docs = child_text_splitter.split_documents([doc])

            # Assign the same unique_id to all child chunks
            for _doc in sub_docs:
                _doc.metadata = doc.metadata.copy()
                child_docs.append(_doc)

        # Save the child chunks as a JSONL file
        #DocumentIndexer.save_docs_to_jsonl(child_docs, "child_documents.jsonl")

        st.markdown("### Children Chunks:")
        st.write(child_docs)

        return child_docs

    