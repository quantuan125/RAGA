import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import Document
from langchain.load import dumps, loads



class PostProcessor:

    @staticmethod
    def display_documents(docs_list):
        formatted_docs = []
        for doc in docs_list:
            if isinstance(doc, Document):
                doc_info = {
                    "Page Content": doc.page_content,
                    # Add other relevant fields here if needed
                }
                formatted_docs.append(doc_info)
            else:
                formatted_docs.append("Not a Document object")

        #st.write(formatted_docs)

    @staticmethod
    def prp_reranking(retrieval_results, question=None):

        if isinstance(retrieval_results[0], Document):
            retrieval_results = [retrieval_results]

        def reciprocal_rank_fusion(results: list[list], k=60):
            fused_scores = {}
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    fused_scores[doc_str] += 1 / (rank + k)
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            return reranked_results
        
        reranked_results = reciprocal_rank_fusion(retrieval_results)

        flat_reranked_results = [doc_score_tuple[0] for doc_score_tuple in reranked_results]


        # st.markdown("### Top Ranked Results:")
        # st.write(flat_reranked_results)
        return flat_reranked_results
    
    @staticmethod
    def contextual_compression(documents, question):
        embeddings = OpenAIEmbeddings()

        # Initialize Text Splitter
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator=". ")

        # Initialize Redundant Filter
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

        # Initialize Relevant Filter
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76, k=30)

        # Create Compressor Pipeline
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        # Compress the documents
        compressed_documents = pipeline_compressor.compress_documents(documents=documents, query=question)
        
        # st.markdown("### Compressed Docs")
        # PostProcessor.display_documents(compressed_documents) 
        return compressed_documents
    
    @staticmethod
    def filter_top_documents(documents, question=None):
        top_n = st.session_state.get('filter_number', 5)  # Default to 5 if not set
        top_docs = documents[:top_n]

        # st.markdown("### Filter Top Docs")
        # PostProcessor.display_documents(top_docs) 
        return top_docs
    
