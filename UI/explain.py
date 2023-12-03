import streamlit as st
import os
import json
import datetime
from langchain.chains.query_constructor.base import AttributeInfo
from pipeline.rag.query_construction import QueryConstructor
from pipeline.rag.vector_search import VectorSearch


class Explain_QT:
    @staticmethod
    def multi_retrieval_query():
        st.markdown(
            '''
            ### Explanation 
            
            This method generates multiple variations of the original question to enhance the retrieval process by covering different phrasings or aspects of the query.

            ### Example

            **Input**: Original Question
            ```
            "How to deposit a cheque issued to an associate in my business into my business account?"
            ```

            **Output**: Multi Generated Questions
            ```
            1. "What is the process for depositing a cheque made out to an associate into my business account?"
            2. "Can you explain how to deposit a cheque issued to an associate into my business account?"
            3. "What are the steps to deposit a cheque made out to an associate into my business account?"
            ```
            '''
        )

    @staticmethod
    def multi_retrieval_query_settings():
        query_count = st.slider("Select the number of queries to generate", 1, 10, 3, 1)
        st.session_state.query_count = query_count

        default_mrq_prompt = """
        You are an AI language model assistant. Your task is to generate {query_count} different versions of the given user question to retrieve relevant documents from a vector database.
        
        Provide these alternative questions separated by newlines.
        Original question: {question}
        """
        mrq_prompt_template = st.text_area("Customize your Multi-Retrieval Query prompt:", value=default_mrq_prompt, height=200)
        st.session_state.mrq_prompt_template = mrq_prompt_template

    @staticmethod
    def rewrite_retrieve_read():
        st.markdown(
            '''
            ### Explanation 
            The Rewrite-Retrieve-Read method improves the retrieval process by first prompting an LLM to generate a more precise and focused search query from a user's initial vague or poorly structured question. This enhanced query is then used to retrieve relevant documents, which are analyzed by the LLM to formulate a comprehensive response.

            ### Example
            **Input**: Original Question
            ```
            "I wanna know about the stuff that makes computers think?"
            ```
            
            **Output**: Rewritten Question
            ```
            "Key principles and technologies in artificial intelligence and machine learning for computer cognition"
            ```
            '''
        )

    def rewrite_retrieve_read_settings():
        pass
    
    @staticmethod
    def query_extractor():
        st.markdown(
            '''
            ### Explanation
            
            The Query Extractor refines the user's natural language query to improve the retrieval of relevant documents from a vector database.
            
            ### Example
            
            **Input**: Original Question
            ```
            "What are the main factors contributing to urban pollution?"
            ```
            
            **Output**: Extracted Question
            ```
            "urban pollution main contributing factors"
            ```
            '''
        )
    
    def query_extractor_settings():
        pass

    @staticmethod
    def step_back_prompting():
        st.markdown(
            '''
            ### Explanation
            
            The Query Extractor refines the user's natural language query to improve the retrieval of relevant documents from a vector database.
            
            ### Example
            
            **Input**: Original Question
            ```
            "What are the main factors contributing to urban pollution?"
            ```
            
            **Output**: Extracted Question
            ```
            "urban pollution main contributing factors"
            ```
            '''
        )

    def step_back_prompting_settings():
        pass

class Explain_QC:

    def self_query_construction():
        st.markdown(
            '''
            **SelfQuery Retrieval:**
            
            This method translates natural language queries into structured queries using metadata filtering. This is particularly effective when dealing with vector databases that include structured data.
            
            **Example:**
            
            Given a natural language query like "What are movies about aliens in the year 1980", SelfQuery Retrieval will decompose the query into a semantic search term and logical conditions for metadata filtering, thus leveraging both semantic and structured data search capabilities.
            '''
        )
        
    def self_query_construction_settings(selected_query_constructor):
        metadata_folder_path = 'json/metadata'
        schema_folder_path = 'json/schema'
        htso_folder_path = 'json/headers_info'
        toc_folder_path = 'json/toc'

        # List available metadata and schema files
        metadata_files = [f for f in os.listdir(metadata_folder_path) if f.endswith('.json')]
        schema_files = [f for f in os.listdir(schema_folder_path) if f.endswith('.json')]
        htso_files = [f for f in os.listdir(htso_folder_path) if f.endswith('.json')]
        toc_files = [f for f in os.listdir(toc_folder_path) if f.endswith('.json')]


        if selected_query_constructor == 'Self-Query Construction':
            st.subheader("Configure Metadata Attribute and Description")
            if 'document_content_description' not in st.session_state:
                st.session_state.document_content_description = "Structured sections of the Danish Building Regulation 2018 document"
            
            st.session_state.document_content_description = st.text_area(
            "Document Content Description",
            value=st.session_state.document_content_description
            )

            # Selectbox for metadata field info files
            selected_metadata_file = st.selectbox("Select Metadata Field Info File:", ['None'] + metadata_files)

            if selected_metadata_file != 'None':
                metadata_info_file_path = os.path.join(metadata_folder_path, selected_metadata_file)
                # Load selected metadata field info
                with open(metadata_info_file_path, 'r') as file:
                    st.session_state.metadata_field_info = json.load(file)
                    st.write(st.session_state.metadata_field_info)
            else:
                st.session_state.metadata_field_info = None

            selected_toc_file = st.selectbox("Select Table of Contents File:", ['None'] + toc_files)

            if selected_toc_file != 'None':
                toc_file_path = os.path.join(toc_folder_path, selected_toc_file)
                # Load selected ToC file
                with open(toc_file_path, 'r') as file:
                    st.session_state.toc_content = json.load(file)
                    st.write(st.session_state.toc_content)
            else:
                st.session_state.toc_content = None

            create_new_metadata_attr = st.checkbox("Create New Metadata Attribute", value=False)
            if create_new_metadata_attr:
                st.subheader("Create Metadata Attribute")

                # Selectbox for index schema files
                selected_schema_file = st.selectbox("Select Index Schema File:", ['None'] + schema_files)

                selected_htso_file = st.selectbox(
                    "Select a 'Headers to Split On' JSON file:",
                    options=['None'] + htso_files
                )

                if st.button("Generate Metadata Field Info"):
                    if selected_schema_file != 'None' and selected_htso_file != 'None':
                        # Load index schema
                        json_schema_file_path = os.path.join(schema_folder_path, selected_schema_file)
                        with open(json_schema_file_path, 'r') as file:
                            index_schema = json.load(file)

                        # Load headers to split on
                        json_htso_file_path = os.path.join(htso_folder_path, selected_htso_file)
                        with open(json_htso_file_path, 'r') as file:
                            headers_info = json.load(file)

                        st.session_state.metadata_field_info = QueryConstructor.build_metadata_field_info(index_schema, headers_info)

                        # Save the generated metadata field info
                        timestamp_info = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        new_metadata_file_path = os.path.join(metadata_folder_path, f"metadata_field_info{timestamp_info}.json")

                        with open(new_metadata_file_path, 'w') as file:
                            json.dump([attr_info.__dict__ for attr_info in st.session_state.metadata_field_info], file, indent=4)
                        
                        st.success(f"Metadata Field Info generated and saved as {new_metadata_file_path}")

                    # Function to add a new attribute
                def add_attribute():
                    next_index = len(st.session_state.metadata_field_info) + 1
                    st.session_state.metadata_field_info.append(
                        AttributeInfo(name=f"Header{next_index}", description=f"Header {next_index} Content", type="string")
                    )

                    # Function to remove the last attribute
                def remove_attribute():
                    if st.session_state.metadata_field_info:
                        st.session_state.metadata_field_info.pop()

                    # Display current attributes and allow editing
                # for index, attribute_info in enumerate(st.session_state.metadata_field_info):
                #     with st.container():
                #         pass
                    # Buttons to add/remove attributes
                col1, col2 = st.columns(2)
                with col1:
                    st.button("Add attribute", on_click=add_attribute, key="add_attribute")
                with col2:
                    st.button("Remove attribute", on_click=remove_attribute, key="remove_attribute")
        
                # Optionally display the current configuration
                # st.write("Current Metadata Fields:")
                # for attribute_info in st.session_state.metadata_field_info:
                #     st.write(f"Name: {attribute_info.name}, Description: {attribute_info.description}")


class Explain_VS:
    def base_retriever():
        st.markdown(
            '''
            ### Explanation 

            The Base Retriever method retrieves documents based on similarity to the input query using vector embeddings.

            ### Example

            **Documents**:
            ```
            "doc1": "Climate change and economic impact.",
            "doc2": "Public health concerns due to climate change.",
            "doc3": "Climate change: A social perspective.",
            "doc4": "Technological solutions to climate change.",
            "doc5": "Policy changes needed to combat climate change.",
            "doc6": "Climate change and its impact on biodiversity.",
            "doc7": "Climate change: The science and models.",
            "doc8": "Global warming: A subset of climate change.",
            "doc9": "How climate change affects daily weather.",
            "doc10": "The history of climate change activism."
            ```

            **Input:** Original Question  
            
            ```
            "impact of climate change"
            ```

            **Output:** Retrieved Documents (top k = 3)
            
            ```
            "doc6": "Climate change and its impact on biodiversity."
            "doc1": "Climate change and economic impact."
            "doc9": "How climate change affects daily weather."
            ```
            '''
        )
    
    def base_retriever_settings():
        top_k_value = st.slider("Select the top k value for retrieval", 1, 20, 3, 1)
        st.session_state.top_k_value = top_k_value

        search_type = st.selectbox("Select the search type", ["similarity", "mmr", "similarity_score_threshold"])
        st.session_state.search_type = search_type

        if search_type == "mmr":
            lambda_mult = st.slider("Select the lambda multiplier for MMR", 0.0, 1.0, 0.5, 0.01)
            st.session_state.lambda_mult = lambda_mult

            fetch_k = st.number_input("Select the fetch k value for MMR", 1, 50, 20, 1)
            st.session_state.fetch_k = fetch_k

        if search_type == "similarity_score_threshold":
            score_threshold = st.slider("Set the similarity score threshold", 0.0, 1.0, 0.8, 0.01)
            st.session_state.score_threshold = score_threshold


    def reranking_retriever():
        st.markdown(
            '''
            ### Explanation 
            This method first retrieves a set of documents similar to the query and then reranks them based on additional criteria to improve relevance.

            ### Example

            **Documents**:
            
            ```
            "doc1": "Climate change and economic impact.",
            "doc2": "Public health concerns due to climate change.",
            "doc3": "Climate change: A social perspective.",
            "doc4": "Technological solutions to climate change.",
            "doc5": "Policy changes needed to combat climate change.",
            "doc6": "Climate change and its impact on biodiversity.",
            "doc7": "Climate change: The science and models.",
            "doc8": "Global warming: A subset of climate change.",
            "doc9": "How climate change affects daily weather.",
            "doc10": "The history of climate change activism."
            ```

            **Input:** Original Question  
            ```
            impact of climate change
            ```

            **Output:** Top Ranked Documents  

            ```
            1. "doc2": "Public health concerns due to climate change.", score: 0.066
            2. "doc3": "Climate change: A social perspective.", score: 0.064
            3. "doc9": "How climate change affects daily weather.", score: 0.048
            4. "doc6": "Climate change and its impact on biodiversity.", score: 0.033
            5. "doc1": "Climate change and economic impact.", score: 0.017
            6. "doc4": "Technological solutions to climate change.", score: 0.017
            7. "doc5": "Policy changes needed to combat climate change.", score: 0.016
            ```
            '''
        )
        
    def reranking_retriever_settings():
        # Slider to set the number of top reranked results
        top_reranked_value = st.slider("Select the number of top reranked results", 1, 20, 5, 1)
        st.session_state.top_reranked_value = top_reranked_value

        # Slider to set the k value for reciprocal rank fusion
        reciprocal_rank_k = st.slider("Select the k value for reciprocal rank fusion", 1, 100, 60, 1)
        st.session_state.reciprocal_rank_k = reciprocal_rank_k

        top_k_value = st.slider("Select the top k value for retrieval", 1, 20, 3, 1)
        st.session_state.top_k_value = top_k_value

        search_type = st.selectbox("Select the search type", ["similarity", "mmr", "similarity_score_threshold"])
        st.session_state.search_type = search_type

        if search_type == "mmr":
            lambda_mult = st.slider("Select the lambda multiplier for MMR", 0.0, 1.0, 0.5, 0.01)
            st.session_state.lambda_mult = lambda_mult

            fetch_k = st.number_input("Select the fetch k value for MMR", 1, 50, 20, 1)
            st.session_state.fetch_k = fetch_k

        if search_type == "similarity_score_threshold":
            score_threshold = st.slider("Set the similarity score threshold", 0.0, 1.0, 0.8, 0.01)
            st.session_state.score_threshold = score_threshold

    def self_query_retriever():
        pass

    def self_query_retriever_settings():
        pass

    def multi_vector_retriever():
        pass

    def multi_vector_retriever_settings():
        inmemorystore_folder = os.path.join(os.getcwd(), "inmemorystore/indexed_documents")
        jsonl_files = [f for f in os.listdir(inmemorystore_folder) if f.endswith('.jsonl')]
        if jsonl_files:
            options = ['None'] + jsonl_files
            selected_file = st.selectbox("Select a JSONL file:", options)

            if selected_file != 'None' and st.button("Load Selected File"):
                file_path = os.path.join(inmemorystore_folder, selected_file)
                st.session_state.inmemorystore = VectorSearch.load_documents_from_jsonl(file_path)
                st.success(f"Loaded documents from {selected_file}")

            # if 'inmemorystore' in st.session_state:
            #     # Display information about the loaded documents
            #     st.markdown("### Loaded Documents:")
            #     st.write(st.session_state.inmemorystore.store)
        else:
            st.write("No JSONL files found in the inmemorystore folder.")
            


class Explain_PP:
    def reranker():
        st.markdown(
            '''
            ### Reranking

            **Explanation:**  
            Reranking involves reordering the initially retrieved documents based on additional relevance criteria or scores.

            **Example:**  
            Given initial retrieval results, reranking might prioritize documents more closely related to specific aspects of the query, resulting in a more refined set of top documents.
            '''
        )
    
    def reranker_settings():
        pass
        


    def contextual_compression():
        st.markdown(
            '''
            ### Contextual Compression

            **Explanation:**  
            This method compresses the retrieved documents by removing redundant information and focusing on the most relevant content related to the query.

            **Example:**  
            From lengthy documents, only the sections or sentences that are most relevant to the query are retained, providing a concise and focused set of documents.
            '''
        )
    
    def contextual_compression_settings():
        pass

    def filter_top_results():
         st.markdown(
            '''
            ### Filter Top Results
            **Explanation:**  
            This process filters the retrieved or reranked documents to only include the top N results, based on a specified number.

            **Example:**  
            If the filter number is set to 5, only the top 5 documents (based on relevance or reranking scores) are selected for further processing.
            '''
        )
    
    def filter_top_results_settings():
        st.session_state.filter_number = st.slider("Select the number of top results to filter", min_value=1, max_value=20, value=5, step=1) 


class Explain_PR:
    def baseline_prompting():
         st.markdown(
            '''
            ### Baseline Prompt
            **Explanation:**  
            Uses a standard template to frame the context and question for generating the response.

            **Example:**  
            ```
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}
            ```
            '''
        )
    
    def baseline_prompting_settings():
        pass

    def custom_prompting():
        custom_example = st.session_state.get('user_custom_prompt', 'No custom prompt defined.')
        st.markdown(
            f"""
            ### Custom Prompting

            **Explanation:**  
            Allows users to define their own prompt template, giving more control over how the AI model generates responses.

            **Example:**  
            
            ```
            {custom_example}

            Context: {{context}}
            
            Question: {{question}}
            ```
            
            """
        )

    def custom_prompting_settings():
        user_defined_prompt = st.text_area("Enter your custom prompt template here:")
        full_custom_prompt = user_defined_prompt + "\n\nContext: {context}\n\nQuestion: {question}"
        st.session_state.user_custom_prompt = user_defined_prompt
        st.session_state.full_custom_prompt = full_custom_prompt
