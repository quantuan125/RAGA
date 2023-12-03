import streamlit as st
import tiktoken


class Pipeline:
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
    def run_retrieval_pipeline(question, query_transformation, query_construction, vector_search, post_processing, prompting):
        pipeline_results = {}

        # If a query transformation method is provided, use it to transform the question
        if query_transformation:
            transformed_questions = query_transformation(question)
        else:
            transformed_questions = question  # If none, just use the original question
        pipeline_results['query_transformation'] = transformed_questions

        if query_construction:
            constructed_query = query_construction(question)
            pipeline_results['query_construction'] = constructed_query

        # Vector search with the transformed questions
        if vector_search:
            retrieval_results = vector_search(transformed_questions)

            if not retrieval_results:
                st.warning("No retrieval results found.")
                return {}

            if isinstance(retrieval_results[0], list):  # Flatten only if it's a list of lists
                retrieval_results = Pipeline.flatten_documents(retrieval_results)
            pipeline_results['vector_search'] = retrieval_results


        # Apply selected PRP methods in order
        for post_processor in post_processing:
            retrieval_results = post_processor(retrieval_results, question)
            if not retrieval_results:
                return st.warning("No retrieval results found after post-processing.")
            
            pipeline_results['post_processing'] = retrieval_results

        # Combine the page_content of each Document into a single string

        if prompting:
            combined_context = Pipeline.create_combined_context(retrieval_results)

            prompt_result = prompting(combined_context, question)

            pipeline_results['prompting'] = prompt_result.get('prompt')
            pipeline_results['answer'] = prompt_result.get('response')

        else:

            pipeline_results['prompting'] = "No prompting method provided."
            pipeline_results['answer'] = None
        
        return pipeline_results
    
    @staticmethod
    def run_ingestion_pipeline(uploaded_file, document_loading, document_splitting, document_processing, document_indexing, document_embedding, vector_databasing):
        if uploaded_file is not None:
            loaded_documents = document_loading(uploaded_file)

            if document_splitting:
                splitted_documents = document_splitting(loaded_documents)
            else:
                splitted_documents = loaded_documents
            
            processed_documents = splitted_documents
            for document_processor in document_processing:
                processed_documents = document_processor(processed_documents)

            #document_chunks = create_final_documents(processed_documents, document_metadata)

            if document_indexing:
                indexed_documents = document_indexing(processed_documents)
            else:
                indexed_documents = processed_documents

            if document_embedding:
                document_embedder = document_embedding()
            else:
                return indexed_documents
            

            if vector_databasing:
                vectorstore = vector_databasing(indexed_documents, document_embedder)
                st.success("Ingestion Complete")
            else:
                raise ValueError("Selected vectorstore type is not supported yet.")

            return vectorstore
        else:
            raise ValueError("No file uploaded. Please upload a file to proceed with indexing.")