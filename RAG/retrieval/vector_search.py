import streamlit as st
from langchain.load import dumps, loads
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
import json
from langchain.vectorstores.redis.schema import RedisModel
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.redis import RedisTranslator
from langchain.chains import create_sql_query_chain



class VectorSearch:
    def __init__(self):
        self.vector_store = st.session_state.vector_store
        self.selected_document = st.session_state.selected_document

    def get_base_retriever_custom(self, top_k=None, search_type=None, lambda_mult=None, fetch_k=None, score_threshold=None):
        top_k = top_k if top_k is not None else st.session_state.get('top_k', 3)
        search_type = search_type if search_type is not None else st.session_state.get('search_type', 'similarity')
        lambda_mult = lambda_mult if lambda_mult is not None else st.session_state.get('lambda_mult', 0.5)
        fetch_k = fetch_k if fetch_k is not None else st.session_state.get('fetch_k', 20)
        score_threshold = score_threshold if score_threshold is not None else st.session_state.get('score_threshold', 0.5)
        
        search_kwargs = {'k': top_k}

        if search_type == 'mmr':
            search_kwargs.update({'lambda_mult': lambda_mult, 'fetch_k': fetch_k})

        if search_type == 'similarity_score_threshold':
            search_kwargs.update({'score_threshold': score_threshold})

        if self.selected_document:
            search_kwargs['filter'] = {'file_name': {'$eq': self.selected_document}}

        base_retriever = st.session_state.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        #st.write(search_type)
        #st.write(search_kwargs)
        return base_retriever
    
    
    def base_retriever(self, questions, **settings):
        base_retriever = self.get_base_retriever_custom(**settings)

        if isinstance(questions, list):
            # Use 'batch' method for a list of queries
            retrieval_results = base_retriever.batch(questions, config={"max_concurrency": 5})
            # Flatten the results as 'batch' returns a list of lists
        else:
            # Use 'invoke' method for a single query string
            retrieval_results = base_retriever.invoke(questions)

        # st.markdown("### Base Retrieval:")
        # st.write(retrieval_results)
        return retrieval_results
    
    def reranking_retriever(self, questions, **settings):

        top_reranked_value = settings.get('top_reranked_value', st.session_state.get('top_reranked_value', 5))
        reciprocal_rank_k = settings.get('reciprocal_rank_k', st.session_state.get('reciprocal_rank_k', 60))
        top_k = settings.get('top_k', st.session_state.get('top_k', 3))

        # Base retrieval process
        base_retriever = self.get_base_retriever_custom(top_k=top_k)

        if isinstance(questions, list):
            # Use 'batch' method for a list of queries
            retriever_map = base_retriever.map()
            retrieval_results = retriever_map.invoke(questions)
            # Flatten the results as 'batch' returns a list of lists
        else:
            # Use 'invoke' method for a single query string
            list_of_retrieval_results = base_retriever.invoke(questions)
            retrieval_results = [list_of_retrieval_results]

        # st.markdown("### Base Retrieval:")
        # st.write(retrieval_results)

        # Reranking process
        fused_scores = {}
        for docs in retrieval_results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + reciprocal_rank_k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Selecting top results and flattening
        top_reranked_results = reranked_results[:top_reranked_value]

        flat_top_reranked_results = [doc_score_tuple[0] for doc_score_tuple in top_reranked_results]

        # st.markdown("### Top Ranked Results:")
        # st.write(top_reranked_results)
        #st.write(flat_top_reranked_results)
        return flat_top_reranked_results
    
    def self_query_retriever(self, question):

        query_constructor = st.session_state.query_constructor
        #st.write(st.session_state.query_constructor)
        index_schema = st.session_state.redis_index_schema
        redis_schema = RedisModel(**index_schema)

        sqr_retriever = SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=st.session_state.vector_store,
            structured_query_translator=RedisTranslator(schema=redis_schema),
        )

        sqr_documents = sqr_retriever.invoke(question)
        
        # st.markdown("### SQR Retrieval Documents:")
        # st.write(sqr_documents)

        return sqr_documents
    
    def multi_vector_retriever(self, questions, **settings):
        mvr_documents_store = settings.get('mvr_documents_store', st.session_state.get('mvr_documents_store', None))

        if not mvr_documents_store:
            raise ValueError("mvr_documents_store is not set. Please configure it before using multi_retriever_query.")

        # Initialize MultiVectorRetriever
        mvr_retriever = MultiVectorRetriever(
            vectorstore=st.session_state.vector_store,
            docstore= mvr_documents_store,
            id_key="unique_id",
        )

        if isinstance(questions, list):
            retrieval_results = mvr_retriever.batch(questions, config={"max_concurrency": 5}) 
        else:
            retrieval_results = mvr_retriever.invoke(questions)

        # st.markdown("### MVR Retrieval Results:")
        # st.write(retrieval_results)

        return retrieval_results
    
    @staticmethod
    def load_documents_from_jsonl(file_path):
        store = InMemoryStore()
        with open(file_path, 'r') as jsonl_file:
            for line in jsonl_file:
                doc_data = json.loads(line)
                doc = Document(page_content=doc_data['page_content'], metadata=doc_data['metadata'])
                store.mset([(doc.metadata['unique_id'], doc)])
        return store

    def sql_retriever(self, question):

        sql_query = st.session_state.sql_query
        sql_result = st.session_state.database.run(sql_query)

        return sql_result
    
