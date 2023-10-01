import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st


class ClientDB:
    def __init__(self, collection_name, load_vector_store=True):
        self.client = chromadb.HttpClient(settings=Settings(allow_reset=True))
        self.collection_name = collection_name
        self.vector_store = None
        if collection_name and load_vector_store:
            self.load_vector_store()

    def load_vector_store(self):
        embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(collection_name=self.collection_name, embedding_function=embeddings, client=self.client)
        st.session_state.vector_store = self.vector_store
        #st.write(st.session_state.vector_store)

    def get_existing_collections(self):
        collections = self.client.list_collections()
        sorted_collection = sorted([col.name for col in collections])
        return sorted_collection

    def reset_client(self):
        self.client.reset()