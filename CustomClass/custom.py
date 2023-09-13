from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Tuple
from langchain.schema.retriever import BaseRetriever, Document

class CustomMultiVectorRetriever(MultiVectorRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Override this method to use similarity_search_with_relevance_scores
        sub_docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )
        
        # Filter by score if needed (you can add additional logic here)
        sub_docs = [doc for doc, score in sub_docs_and_scores if score >= 0.8]
        
        # ... (rest of the method stays the same)
        ids = []
        for d in sub_docs:
            if d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        
        return [d for d in docs if d is not None]