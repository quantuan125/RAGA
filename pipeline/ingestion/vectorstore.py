import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.redis import Redis
import uuid

class VectorStore:

    def build_chroma_vectorstore(document_chunks, document_embedder):
        chroma_client = st.session_state.client_db.client
        chroma_collection_name = st.session_state.collection_name
        
        # Use the ids that were passed to the function, which already include the file name
        use_unique_id = all('unique_id' in doc.metadata for doc in document_chunks)

        # If 'unique_id' is present in all documents, use it as the id
        ids = [doc.metadata['unique_id'] for doc in document_chunks] if use_unique_id else None

        unique_ids = [doc.metadata.get('unique_id') for doc in document_chunks]
        if len(unique_ids) != len(set(unique_ids)):
            # Generate new IDs combining filename and a UUID
            ids = [f"{doc.metadata.get('file_name', 'unknown')}_{uuid.uuid4()}" for doc in document_chunks]
        else:
            ids = unique_ids

        vectorstore = Chroma.from_documents(
            documents=document_chunks, 
            embedding=document_embedder, 
            ids=ids,  # This list is now passed in as a parameter
            client=chroma_client, 
            collection_name=chroma_collection_name
        )

        st.markdown("### Build Vectorstore:")
        #st.write(vectorstore)
        return vectorstore
    
    def build_redis_vectorstore(document_chunks, document_embedder):
        redis_url = st.session_state.get('redis_url', 'redis://localhost:9000')
        redis_index_name = st.session_state.get('redis_index_name', 'user')
        
        redis_vectorstore = Redis.from_documents(
            document_chunks,
            document_embedder,
            redis_url=redis_url,
            index_name = redis_index_name,
        )

        return redis_vectorstore