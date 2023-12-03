import streamlit as st
from langchain.embeddings import OpenAIEmbeddings


class DocumentEmbedder:
    
    @staticmethod
    def openai_embedding_model():
        return OpenAIEmbeddings()