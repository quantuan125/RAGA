import streamlit as st
import os
import json
from pipeline.pipeline import Retrieval_Pipeline, Ingestion_Pipeline

class ConfigSettings:
    def get_function_name(func):
            """Return the name of the function or 'None' if the function is None."""
            return func.__name__ if func is not None else 'None'


class RetrievalConfigSettings:
        
    def get_query_transformation_settings(function_name):
        if function_name == 'multi_retrieval_query':
            return {
                'query_count': st.session_state.query_count,
                'mrq_prompt_template': st.session_state.mrq_prompt_template
            }
        return None  
    
    @staticmethod
    def get_query_construction_settings(function_name):
        if function_name == 'self_query_constructor':
            return {
                'document_content_description': st.session_state.get('document_content_description', None),
                'metadata_field_info': st.session_state.get('metadata_field_info', None),
                'toc_content': st.session_state.get('toc_content', None)
            }
        return None
    
    def get_vector_search_settings(function_name):
        if function_name == 'base_retriever':
            return {
                'top_k': st.session_state.get('top_k_value', 3),
                'search_type': st.session_state.get('search_type', 'similarity'),
                'lambda_mult': st.session_state.get('lambda_mult', 0.5),
                'fetch_k': st.session_state.get('fetch_k', 20),
                'score_threshold': st.session_state.get('score_threshold', 0.5)
            }
        elif function_name == 'reranking_retriever':
            return {
            'top_reranked_value': st.session_state.get('top_reranked_value', 5),
            'reciprocal_rank_k': st.session_state.get('reciprocal_rank_k', 60),
            'top_k': st.session_state.get('top_k_value', 3)
        }
        # elif function_name == 'multi_vector_retriever':
        #     return {
        #         'mvr_documents_store': st.session_state.get('mvr_documents_store', None)
        #     }
        return None
        

    def get_post_processing_settings(function_name):
        if function_name == 'filter_top_documents':
            return {'top_filter_n': st.session_state.top_filter_n}
        # Add other conditions for different functions
        return None
    
    def get_prompting_settings(function_name):
        if function_name == 'custom_prompt':
            return {
                'custom_template': st.session_state.get('full_custom_prompt', None)
            }
        # Add other conditions for different prompting functions if necessary
        return None
    
    @staticmethod
    def get_baseline_json_config():
        baseline_json_config = {
                "query_transformation": {
                    "function": "None",
                    "settings": {}
                },
                "query_construction": {
                    "function": "None",
                    "settings": {}
                },
                "vector_search": {
                    "function": "base_retriever",
                    "settings": {}
                },
                "post_processing": {
                    "functions": []
                },
                "prompting": {
                    "function": "baseline_prompt",
                    "settings": {}
                }
            }
        return baseline_json_config
    
    @staticmethod
    def create_fss_config(query_transformation_function, query_construction_function, vector_search_function, post_processing_function, prompting_function):

        fss_config = {
            'query_transformation_function': query_transformation_function,
            'query_construction_function': query_construction_function,
            'vector_search_function': vector_search_function,
            'post_processing_functions': post_processing_function,
            'prompting_function': prompting_function
        }

        return fss_config
    
    @staticmethod
    def convert_fss_to_json_config(query_transformation_function, query_construction_function, vector_search_function, post_processing_function, prompting_function):
        json_config = {
            'query_transformation': {
                'function': ConfigSettings.get_function_name(query_transformation_function),
                'settings': RetrievalConfigSettings.get_query_transformation_settings(ConfigSettings.get_function_name(query_transformation_function))
            },
            'query_construction': {
                'function': ConfigSettings.get_function_name(query_construction_function),
                'settings': RetrievalConfigSettings.get_query_construction_settings(ConfigSettings.get_function_name(query_construction_function))
            },
            'vector_search': {
                'function': ConfigSettings.get_function_name(vector_search_function),
                'settings': RetrievalConfigSettings.get_vector_search_settings(ConfigSettings.get_function_name(vector_search_function))
            },
            'post_processing': {
                'functions': [
                    {
                        'function': ConfigSettings.get_function_name(func),
                        'settings': RetrievalConfigSettings.get_post_processing_settings(ConfigSettings.get_function_name(func))
                    } for func in post_processing_function
                ]
            },
            'prompting': {
                'function': ConfigSettings.get_function_name(prompting_function),
                'settings': RetrievalConfigSettings.get_prompting_settings(ConfigSettings.get_function_name(prompting_function))
            },
        }
        return json_config
    
    @staticmethod
    def mapping_retrieval_config(json_config):
        """
        Convert a JSON config to FSS format. The config must be a dictionary.
        """
        if not isinstance(json_config, dict):
            raise ValueError("Invalid configuration input. Must be a JSON dict.")
        
        fss_config = {
            'query_transformation_function': Retrieval_Pipeline.get_retrieval_pipeline_function('query_transformation', json_config.get('query_transformation', 'None')),
            'query_construction_function': Retrieval_Pipeline.get_retrieval_pipeline_function('query_construction', json_config.get('query_construction', 'None')),
            'vector_search_function': Retrieval_Pipeline.get_retrieval_pipeline_function('vector_search', json_config.get('vector_search', 'None')),
            'post_processing_functions': [Retrieval_Pipeline.get_retrieval_pipeline_function('post_processing', func_info) for func_info in json_config.get('post_processing', {}).get('functions', []) if func_info.get('function') != 'None'],
            'prompting_function': Retrieval_Pipeline.get_retrieval_pipeline_function('prompting', json_config.get('prompting', 'None'))
        }

        return fss_config
    
    @staticmethod
    def run_retrieval_pipeline_with_config(user_question, config_settings):
        return Retrieval_Pipeline.retrieval_pipeline(
            user_question,
            query_transformation=config_settings['query_transformation_function'],
            query_construction=config_settings['query_construction_function'],
            vector_search=config_settings['vector_search_function'],
            post_processing=config_settings['post_processing_functions'],
            prompting=config_settings['prompting_function']
        )
    
class IngestionConfigSettings:
    
    def get_document_loading_settings(function_name):
        pass

    def get_document_splitting_settings(function_name):
        pass

    def get_document_processing_settings(function_name):
        pass

    def get_document_indexing_settings(function_name):
        pass

    def get_document_embedding_settings(function_name):
        pass

    def get_vector_database_settings(function_name):
        pass

    @staticmethod
    def get_baseline_json_config():
        baseline_json_config = {
            "document_loading": {
                "function": "document_loader_langchain",
                "settings": {}
            },
            "document_splitting": {
                "function": "recursive_splitter",
                "settings": {}
            },
            "document_processing": {
                "functions": [
                    {
                        "function": "clean_chunks_content",
                        "settings": {}
                    },
                    {
                        "function": "customize_document_metadata",
                        "settings": {}
                    },
                    {
                        "function": "filter_short_documents",
                        "settings": {}
                    }
                ]
            },
            "document_indexing": {
                "function": "None",
                "settings": {}
            },
            "document_embedding": {
                "function": "None",
                "settings": {}
            },
            "vector_database": {
                "function": "None",
                "settings": {}
            }
        }
        return baseline_json_config

    @staticmethod
    def create_fss_config(document_loading_function, document_splitting_function, document_processing_function, document_indexing_function, document_embedding_function, vector_database_function):
        """
        Create an FSS config for Ingestion settings from the provided function objects.
        """
        fss_config = {
            'document_loading_function': document_loading_function,
            'document_splitting_function': document_splitting_function,
            'document_processing_function': document_processing_function,
            'document_indexing_function': document_indexing_function,
            'document_embedding_function': document_embedding_function,
            'vector_database_function': vector_database_function
        }

        return fss_config

    def convert_fss_to_json_config(document_loading_function, document_splitting_function, document_processing_function, document_indexing_function, document_embedding_function, vector_database_function):
        json_config = {
            'document_loading': {
                'function': ConfigSettings.get_function_name(document_loading_function),
                'settings': IngestionConfigSettings.get_document_loading_settings(ConfigSettings.get_function_name(document_loading_function)) 
            },
            'document_splitting': {
                'function': ConfigSettings.get_function_name(document_splitting_function),
                'settings': IngestionConfigSettings.get_document_splitting_settings(ConfigSettings.get_function_name(document_splitting_function))  
            },
            'document_processing': {
                'functions': [
                    {
                        'function': ConfigSettings.get_function_name(func),
                        'settings': IngestionConfigSettings.get_document_processing_settings(ConfigSettings.get_function_name(func))  
                    } for func in document_processing_function
                ]
            },
            'document_indexing': {
                'function': ConfigSettings.get_function_name(document_indexing_function),
                'settings': IngestionConfigSettings.get_document_indexing_settings(ConfigSettings.get_function_name(document_indexing_function))
            },
            'document_embedding': {
                'function': ConfigSettings.get_function_name(document_embedding_function),
                'settings': IngestionConfigSettings.get_document_embedding_settings(ConfigSettings.get_function_name(document_embedding_function))
            },
            'vector_database': {
                'function': ConfigSettings.get_function_name(vector_database_function),
                'settings': IngestionConfigSettings.get_vector_database_settings(ConfigSettings.get_function_name(vector_database_function))
            },
        }
        return json_config
    
    @staticmethod
    def mapping_ingestion_config(json_config):
        """
        Convert a JSON config to FSS format. The config must be a dictionary.
        """
        if not isinstance(json_config, dict):
            raise ValueError("Invalid configuration input. Must be a JSON dict.")


        fss_config = {
            'document_loading_function': Ingestion_Pipeline.get_ingestion_pipeline_function('document_loading', json_config.get('document_loading', 'None')),
            'document_splitting_function': Ingestion_Pipeline.get_ingestion_pipeline_function('document_splitting', json_config.get('document_splitting', 'None')),
            'document_processing_function': [Ingestion_Pipeline.get_ingestion_pipeline_function('document_processing', func_info) for func_info in json_config.get('document_processing', {}).get('functions', []) if func_info.get('function') != 'None'],
            'document_indexing_function': Ingestion_Pipeline.get_ingestion_pipeline_function('document_indexing', json_config.get('document_indexing', 'None')),
            'document_embedding_function': Ingestion_Pipeline.get_ingestion_pipeline_function('document_embedding', json_config.get('document_embedding', 'None')),
            'vector_database_function': Ingestion_Pipeline.get_ingestion_pipeline_function('vector_database', json_config.get('vector_database', 'None'))
        }

        return fss_config
    
    @staticmethod
    def run_ingestion_pipeline_with_config(uploaded_file, config_settings):
        return Ingestion_Pipeline.ingestion_pipeline(
            uploaded_file,
            document_loading=config_settings['document_loading_function'],
            document_splitting=config_settings['document_splitting_function'],
            document_processing=config_settings['document_processing_function'],
            document_indexing=config_settings['document_indexing_function'],
            document_embedding=config_settings['document_embedding_function'],
            vector_database=config_settings['vector_database_function']
        )
    

class JSONConfigHandler:

    @staticmethod
    def save_json_config(config_dict, file_path):
        # Convert dict to JSON string
        json_string = json.dumps(config_dict, indent=4)
        # Write to file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
        with open(file_path, 'w') as f:
            f.write(json_string)

    @staticmethod
    def load_json_config(config_folder_path, filename):
        """Load a JSON configuration file from the specified path and return the settings."""
        with open(os.path.join(config_folder_path, filename), 'r') as f:
            return json.load(f)    
    
    @staticmethod
    def list_json_config_files(config_folder_path):
        """List all JSON configuration files in the specified config directory."""
        files = [f for f in os.listdir(config_folder_path) if f.endswith('.json')]
        return ['Default'] + files  # Include 'Default' option
        
    
        


   
    
    


    

        
