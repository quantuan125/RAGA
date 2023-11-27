import streamlit as st
from langchain.load import dumps, loads
from langchain.chat_models import ChatOpenAI
from agent.tools import DatabaseTool


class VectorSearch:
    def __init__(self):
        self.vector_store = st.session_state.vector_store
        self.selected_document = st.session_state.selected_document
        self.retriever = self.get_base_retriever_custom()

    def get_base_retriever_custom(self):
        top_k = st.session_state.get('top_k_value', 3)
        search_type = st.session_state.get('search_type', 'similarity')
        search_kwargs = {'k': top_k}

        if search_type == 'mmr':
            lambda_mult = st.session_state.get('lambda_mult', 0.5)
            fetch_k = st.session_state.get('fetch_k', 20)
            search_kwargs.update({'lambda_mult': lambda_mult, 'fetch_k': fetch_k})

        if search_type == 'similarity_score_threshold':
            score_threshold = st.session_state.get('score_threshold', 0.5)
            search_kwargs.update({'score_threshold': score_threshold})

        if self.selected_document:
            search_kwargs['filter'] = {'file_name': {'$eq': self.selected_document}}

        base_retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        #st.write(search_type)
        #st.write(search_kwargs)
        return base_retriever
    
    def base_retriever(self, generated_queries, retriever=None):
        base_retriever = retriever if retriever is not None else self.retriever

        if isinstance(generated_queries, list):
            # Use 'batch' method for a list of queries
            retrieval_results = base_retriever.batch(generated_queries, config={"max_concurrency": 5})
            # Flatten the results as 'batch' returns a list of lists
        else:
            # Use 'invoke' method for a single query string
            retrieval_results = base_retriever.invoke(generated_queries)

        st.markdown("### Base Retrieval:")
        st.write(retrieval_results)
        return retrieval_results
    
    def reranking_retriever(self, generated_queries, retriever=None):
    # Retrieve session state values or use default
        top_reranked_value = st.session_state.get('top_reranked_value', 5)
        reciprocal_rank_k = st.session_state.get('reciprocal_rank_k', 60)

        # Base retrieval process
        base_retriever = retriever if retriever is not None else self.retriever

        retriever_map = base_retriever.map()
        retrieval_results = retriever_map.invoke(generated_queries)

        st.markdown("### Base Retrieval:")
        st.write(retrieval_results)

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

        st.markdown("### Top Ranked Results:")
        st.write(top_reranked_results)
        return flat_top_reranked_results
    
    def self_query_retriever(self, question, sqr_retriever):

        structured_query = sqr_retriever.query_constructor.invoke({"query": question})
        
        st.markdown("### Structured Request Output:")
        st.json({"query": structured_query.query, "filter": str(structured_query.filter)})
        
        # Prepare and execute the query
        new_query, search_kwargs = sqr_retriever._prepare_query(question, structured_query)
        sqr_documents = sqr_retriever._get_docs_with_query(new_query, search_kwargs)
        
        st.markdown("### SQR Retrieval Documents:")
        st.write(sqr_documents)

        return sqr_documents