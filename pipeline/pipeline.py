import streamlit as st
import tiktoken
from pipeline.rag.query_transformation import QueryTransformer
from pipeline.rag.query_construction import QueryConstructor
from pipeline.rag.vector_search import VectorSearch
from pipeline.rag.post_processing import PostProcessor
from pipeline.rag.prompting import Prompting
from pipeline.ingestion.document_loader import DocumentLoader
from pipeline.ingestion.document_splitter import DocumentSplitter
from pipeline.ingestion.document_processor import DocumentProcessor
from pipeline.ingestion.document_indexer import DocumentIndexer
from pipeline.ingestion.document_embedder import DocumentEmbedder
from pipeline.ingestion.vectorstore import VectorStore


class Retrieval_Pipeline:
    query_transformation_instance = None
    query_construction_instance = None
    vector_search_instance = None
    post_processing_instance = None
    prompting_instance =None
    retrieval_pipeline_function_mappings = {}

    @classmethod
    def initialize_retrieval_instances_and_mappings(cls):
        if not cls.query_transformation_instance:
            cls.query_transformation_instance = QueryTransformer()
        if not cls.query_construction_instance:
            cls.query_construction_instance = QueryConstructor()
        if not cls.vector_search_instance:
            cls.vector_search_instance = VectorSearch()
        if not cls.post_processing_instance: 
            cls.post_processing_instance = PostProcessor()
        if not cls.prompting_instance:
            cls.prompting_instance = Prompting()

        if not cls.retrieval_pipeline_function_mappings:
            # Define the function mappings
            cls.retrieval_pipeline_function_mappings = {
                'query_transformation': {
                    'None': None,
                    'multi_retrieval_query': cls.query_transformation_instance.multi_retrieval_query,
                    'rewrite_retrieve_read': cls.query_transformation_instance.rewrite_retrieve_read,
                    'query_extractor': cls.query_transformation_instance.query_extractor,
                    'step_back_prompting': cls.query_transformation_instance.step_back_prompting,
                },

                'query_construction': {
                    'None': None,
                    'self_query_constructor': cls.query_construction_instance.self_query_constructor,
                },

                'vector_search': {
                    'None': None,
                    'base_retriever': cls.vector_search_instance.base_retriever,
                    'reranking_retriever': cls.vector_search_instance.reranking_retriever,
                    'self_query_retriever':  cls.vector_search_instance.self_query_retriever,
                    'multi_vector_retriever': cls.vector_search_instance.multi_vector_retriever,
                },

                'post_processing': {
                    'None': None,
                    'contextual_compression': cls.post_processing_instance.contextual_compression,
                    'reranking': cls.post_processing_instance.reranking,
                    'filter_top_documents': cls.post_processing_instance.filter_top_documents,
                },

                'prompting': {
                    'None': None,
                    'baseline_prompt': cls.prompting_instance.baseline_prompt,
                    'custom_prompt': cls.prompting_instance.custom_prompt,
                    'step_back_prompt': cls.prompting_instance.step_back_prompt,
                },
            }
            
    @classmethod
    def check_and_initialize_vector_search(cls):
        # Only reinitialize if vector_store has been set or changed
        if 'vector_store' in st.session_state and (not cls.vector_search_instance or cls.vector_search_instance.vector_store != st.session_state.vector_store):
            cls.vector_search_instance = VectorSearch()

    @classmethod
    def get_retrieval_pipeline_function(cls, step_name, func_info):
        """
        Get the function based on the step name and function name.
        The step_name is used to select the correct class instance,
        and func_name is used to get the specific method.
        """
        cls.initialize_retrieval_instances_and_mappings()

        if isinstance(func_info, str):
            func_name = func_info
            settings = {}
        elif isinstance(func_info, dict):
            func_name = func_info.get('function', 'None')
            settings = func_info.get('settings', {})
        else:
            return None

        # Retrieve the mapping for the specific pipeline step
        function_mapping = cls.retrieval_pipeline_function_mappings.get(step_name)
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
    def create_combined_context(retrieval_results, max_tokens=15500, model_name='gpt-3.5-turbo'):
        def truncate_to_token_limit(text, max_tokens, model_name):
            # Retrieve the correct encoding for the model
            encoding = tiktoken.encoding_for_model(model_name)

            # Encode the text and get the number of tokens
            tokens = encoding.encode(text)
            num_tokens = len(tokens)

            # If the number of tokens is within the limit, return the text as is
            if num_tokens <= max_tokens:
                return text
            
            # If the number of tokens exceeds the limit, truncate
            truncated_text = encoding.decode(tokens[:max_tokens])
            return truncated_text
        
        # Combine the page_content of each Document into a single string
        #st.write(retrieval_results)
        combined_context = "\n\n".join([doc.page_content for doc in retrieval_results])

        # Ensure the combined context is within the token limit
        combined_context = truncate_to_token_limit(combined_context, max_tokens, model_name)

        #st.markdown("### Combined Text:")
        #st.write(combined_context)
        return combined_context

    @staticmethod
    def flatten_documents(nested_documents):
        """Flatten a list of lists of Document objects into a flat list of Document objects."""
        flatten_documents = [doc for sublist in nested_documents for doc in sublist]
        #st.write(flatten_documents)
        return flatten_documents
    
    @staticmethod
    def retrieval_pipeline(question, query_transformation, query_construction, vector_search, post_processing, prompting):
        pipeline_results = {}

        # If a query transformation method is provided, use it to transform the question
        if query_transformation:
            transformed_questions = query_transformation(question)
        else:
            transformed_questions = question  # If none, just use the original question
        pipeline_results['query_transformation'] = transformed_questions

        if query_construction:
            constructed_results = query_construction(question)
            pipeline_results['query_construction'] = constructed_results

        # Vector search with the transformed questions
        if vector_search:
            retrieval_results = vector_search(transformed_questions)

            if not retrieval_results:
                st.warning("No retrieval results found.")
                return {}

            if isinstance(retrieval_results[0], list):  # Flatten only if it's a list of lists
                retrieval_results = Retrieval_Pipeline.flatten_documents(retrieval_results)
            pipeline_results['vector_search'] = retrieval_results


        pipeline_results['post_processing'] = {}
        # Apply selected PRP methods in order
        for post_processor in post_processing:
            method_name = post_processor.__name__  # Get the name of the method
            retrieval_results = post_processor(retrieval_results, question)

            if not retrieval_results:
                return st.warning(f"No retrieval results found after {method_name}.")

            # Store the results of each post-processing method separately
            pipeline_results['post_processing'][method_name] = retrieval_results

        # Combine the page_content of each Document into a single string

        if prompting:
            combined_context = Retrieval_Pipeline.create_combined_context(retrieval_results)

            prompt_result = prompting(combined_context, question)

            pipeline_results['prompting'] = prompt_result.get('prompt')
            pipeline_results['answer'] = prompt_result.get('response')

        else:

            pipeline_results['prompting'] = "No prompting method provided."
            pipeline_results['answer'] = None
        
        return pipeline_results
    
        

class Ingestion_Pipeline:
    document_loading_instance = None
    document_splitting_instance = None
    document_processing_instance = None
    document_indexing_instance = None
    document_embedding_instance = None
    vector_database_instance = None 
    ingestion_pipeline_function_mappings = {}

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
                'document_loader_unstructured': cls.document_loading_instance.document_loader_unstructured,
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
    

    
