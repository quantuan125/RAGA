import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
from utility.authy import Login
import os

class ClientDB:
    def __init__(self, username, collection_name, load_vector_store=True):
        #user_port = Login.get_port_for_user(username)
        #if not user_port:
            #raise ValueError(f"No server port found for user {username}")
        
        if username == "admin":
            auth_credentials = os.getenv('ADMIN_AUTHENTICATION')
        else:
            auth_credentials = username 

        # Set up client with appropriate credentials and server URL
        self.client = chromadb.HttpClient(
            host=os.getenv("CHROMA_SERVER_HOST", "default_host"),
            port=8000,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                chroma_client_auth_credentials=auth_credentials,
                allow_reset=True
            )
        )
        #st.write(self.client)
        self.collection_name = collection_name
        self.vector_store = None
        if collection_name and load_vector_store:
            self.load_vector_store()

    def load_vector_store(self):
        embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(collection_name=self.collection_name, embedding_function=embeddings, client=self.client)
        st.session_state.vector_store = self.vector_store
        #st.write(st.session_state.vector_store)

    def get_user_specific_collections(self):
        collections = self.client.list_collections()
        user_collections = sorted([col.name for col in collections if col.name.startswith(f"{st.session_state.username}-")])
        return user_collections
    
    def get_all_sorted_collections(self):
        all_collections_objects = self.client.list_collections()
        sorted_collections_objects = sorted(
            all_collections_objects, 
            key=lambda collection: collection.name.lower()
        )
        return sorted_collections_objects
    
    def get_user_sorted_collections(self, username):
        all_collections_objects = self.client.list_collections()
        user_collections_objects = [col for col in all_collections_objects if col.name.startswith(f"{username}-")]
        user_sorted_collections_objects = sorted(
            user_collections_objects, 
            key=lambda collection: collection.name.lower()
        )
        return user_sorted_collections_objects

    def reset_client(self):
        self.client.reset()

    def validate_access(self, collection_name):
        return collection_name.startswith(f"{st.session_state.username}-")

    