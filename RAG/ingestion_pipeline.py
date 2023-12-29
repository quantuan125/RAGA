import streamlit as st
from RAG.ingestion.document_loader import DocumentLoader
from RAG.ingestion.document_splitter import DocumentSplitter
from RAG.ingestion.document_processor import DocumentProcessor
from RAG.ingestion.document_indexer import DocumentIndexer
from RAG.ingestion.document_embedder import DocumentEmbedder
from RAG.ingestion.vectorstore import VectorStore


class Ingestion_Pipeline:
    document_loading_instance = None
    document_splitting_instance = None
    document_processing_instance = None
    document_indexing_instance = None
    document_embedding_instance = None
    vector_database_instance = None 
    ingestion_pipeline_function_mappings = {}

    current_step = 0
    steps = ['document_loading', 'document_splitting']
    loaded_documents = None
    splitted_documents = None

    @classmethod
    def initialize_ingestion_instances_and_mappings(cls):
        if not cls.document_loading_instance:
            cls.document_loading_instance = DocumentLoader()
        if not cls.document_splitting_instance:
            cls.document_splitting_instance = DocumentSplitter()
        if not cls.document_processing_instance:
            cls.document_processing_instance = DocumentProcessor()
        if not cls.document_indexing_instance:
            cls.document_indexing_instance = DocumentIndexer()
        if not cls.document_embedding_instance: 
            cls.document_embedding_instance = DocumentEmbedder()
        if not cls.vector_database_instance:
            cls.vector_database_instance = VectorStore()

        cls.ingestion_pipeline_function_mappings = {
            'document_loading': {
                'None': None,
                'document_loader_langchain': cls.document_loading_instance.document_loader_langchain,
                # 'document_loader_unstructured': cls.document_loading_instance.document_loader_unstructured,
            },
            'document_splitting': {
                'None': None,
                'recursive_splitter': cls.document_splitting_instance.recursive_splitter,
                'character_splitter': cls.document_splitting_instance.character_splitter,
                'markdown_header_splitter': cls.document_splitting_instance.markdown_header_splitter,
            },
            'document_processing': {
                'clean_chunks_content': cls.document_processing_instance.clean_chunks_content,
                'customize_document_metadata': cls.document_processing_instance.customize_document_metadata,
                'filter_short_documents': cls.document_processing_instance.filter_short_documents,
                'build_index_schema': cls.document_processing_instance.build_index_schema,
                'build_toc_from_documents': cls.document_processing_instance.build_toc_from_documents,
            },
            'document_indexing': {
                'None': None,
                'summary_indexing': cls.document_indexing_instance.summary_indexing,
                'parent_document_indexing': cls.document_indexing_instance.parent_document_indexing,
            },
            'document_embedding': {
                'None': None,
                'openai_embedding_model': cls.document_embedding_instance.openai_embedding_model,
            },
            'vector_database': {
                'chroma': cls.vector_database_instance.build_chroma_vectorstore,
                'redis': cls.vector_database_instance.build_redis_vectorstore,
                'None': None,
            },
        }

    @classmethod
    def get_ingestion_pipeline_function(cls, step_name, func_info):
        """
        Get the function based on the step name and function name.
        The step_name is used to select the correct class instance,
        and func_name is used to get the specific method.
        """
        cls.initialize_ingestion_instances_and_mappings()

        if isinstance(func_info, str):
            func_name = func_info
            settings = {}
        elif isinstance(func_info, dict):
            func_name = func_info.get('function', 'None')
            settings = func_info.get('settings', {})
        else:
            return None

        # Retrieve the mapping for the specific pipeline step
        function_mapping = cls.ingestion_pipeline_function_mappings.get(step_name)
        if not function_mapping:
            return None
        
        func = function_mapping.get(func_name)
        if not func:
            return None

        # If there are settings, we assume the function can accept them as parameters
        if settings:
            return lambda *args, **kwargs: func(*args, **kwargs, **settings)
        else:
            return func

    @staticmethod
    def ingestion_pipeline(uploaded_file, document_loading, document_splitting, document_processing, document_indexing, document_embedding, vector_database):
        ingestion_results = {}
        
        if uploaded_file is not None:
            loaded_documents = document_loading(uploaded_file)
            ingestion_results['document_loading'] = loaded_documents

            if document_splitting:
                splitted_documents = document_splitting(loaded_documents)
            else:
                splitted_documents = loaded_documents
            ingestion_results['document_splitting'] = splitted_documents
            
            processed_documents = splitted_documents
            ingestion_results['document_processing'] = {}
            for document_processor in document_processing:
                method_name = document_processor.__name__
                processed_documents = document_processor(processed_documents)
                ingestion_results['document_processing'][method_name] = processed_documents

            #document_chunks = create_final_documents(processed_documents, document_metadata)

            if document_indexing:
                indexed_documents = document_indexing(processed_documents)
                ingestion_results['document_indexing'] = indexed_documents
            else:
                indexed_documents = processed_documents
                ingestion_results['document_indexing'] = None
            

            if document_embedding:
                document_embedder = document_embedding()
                ingestion_results['document_embedding'] = document_embedding.__name__
            else:
                ingestion_results['document_embedding'] = None
            

            if vector_database:
                vectorstore = vector_database(indexed_documents, document_embedder)
                ingestion_results['vector_database'] = vectorstore
                st.success("Ingestion Complete")
            else:
                ingestion_results['vector_database'] = None
                return ingestion_results

            #st.write(ingestion_results)
            return ingestion_results
        else:
            raise ValueError("No file uploaded. Please upload a file to proceed with indexing.")
        

    # @classmethod
    # def start_ingestion_pipeline(cls, uploaded_file):
    #     cls.current_step = 0
    #     st.session_state['current_step'] = cls.current_step
    #     cls.execute_current_step(uploaded_file)

    # @classmethod
    # def execute_current_step(cls, uploaded_file=None):
    #     loading_method = st.session_state.get('document_loading_function', None)
    #     splitting_method = st.session_state.get('document_splitting_function', None)

    #     if st.session_state['current_step'] == 0 and loading_method:  # Document Loading
    #         st.session_state['loaded_documents'] = loading_method(uploaded_file)
    #     elif st.session_state['current_step'] == 1 and splitting_method:  # Document Splitting
    #         if splitting_method:
    #             st.session_state['splitted_documents'] = splitting_method(st.session_state['loaded_documents'])
    #         else:
    #             st.session_state['splitted_documents'] = st.session_state['loaded_documents']


    # @classmethod
    # def go_to_next_step(cls):
    #     if st.session_state['current_step'] < len(cls.steps) - 1:
    #         st.session_state['current_step'] += 1
    #         cls.execute_current_step()

    # @classmethod
    # def go_to_previous_step(cls):
    #     if st.session_state['current_step'] > 0:
    #         st.session_state['current_step'] -= 1
    #         cls.execute_current_step()

    # @classmethod
    # def get_current_step_results(cls):
    #     if st.session_state['current_step'] == 0:
    #         return st.session_state.get('loaded_documents', [])
    #     elif st.session_state['current_step'] == 1:
    #         return st.session_state.get('splitted_documents', [])

    
    