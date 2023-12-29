import streamlit as st
import tiktoken
from typing import List
from langchain.schema import Document
from RAG.retrieval.query_transformation import QueryTransformer
from RAG.retrieval.query_construction import QueryConstructor
from RAG.retrieval.vector_search import VectorSearch
from RAG.retrieval.post_processing import PostProcessor
from RAG.retrieval.prompting import Prompting



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
                    'sql_query_constructor': cls.query_construction_instance.sql_query_constructor,
                    'sql_semantic_query_constructor': cls.query_construction_instance.sql_semantic_query_constructor
                },

                'vector_search': {
                    'None': None,
                    'base_retriever': cls.vector_search_instance.base_retriever,
                    'reranking_retriever': cls.vector_search_instance.reranking_retriever,
                    'self_query_retriever':  cls.vector_search_instance.self_query_retriever,
                    'multi_vector_retriever': cls.vector_search_instance.multi_vector_retriever,
                    'sql_retriever': cls.vector_search_instance.sql_retriever
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
                    'sql_prompt': cls.prompting_instance.sql_prompt,
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
        if retrieval_results and isinstance(retrieval_results, List) and all(isinstance(doc, Document) for doc in retrieval_results):
            combined_context = "\n\n".join([doc.page_content for doc in retrieval_results])

            # Ensure the combined context is within the token limit
            combined_context = truncate_to_token_limit(combined_context, max_tokens, model_name)

        else:
            combined_context = retrieval_results

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
    
        



    
