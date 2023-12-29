import streamlit as st
import os
import json
import datetime
from langchain.chains.query_constructor.base import AttributeInfo
from RAG.retrieval.query_construction import QueryConstructor
from RAG.retrieval.vector_search import VectorSearch


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
            
            Step-Back Prompting simplifies detailed questions into broader queries, aiding LLMs to grasp essential concepts without getting overwhelmed by specifics. This technique generates high-level questions that capture the core idea of the original query, thereby enhancing the model's retrieval and response accuracy for complex questions.

            ### Example
            
            **Input**: Original Question
            ```
            "Which team did Cristiano Ronaldo play for from 2007 to 2008"
            ```

            **Output**: Step-Back Question
            ```
            "Which teams did Cristiano Ronaldo play for in his career?"
            ```
            '''
        )


    def step_back_prompting_settings():
        pass

class Explain_QC:

    def self_query_construction():
        st.markdown(
            '''
            ### Explanation
            
            The Self-Query Construction method transforms natural language questions into structured queries that align with the document's metadata, enhancing retrieval from vector databases. It isolates the semantic query and constructs a metadata filter based on logical conditions. This technique is critical for producing precise structured requests that guide the LLM in sourcing relevant information effectively.

            ### Example
            
            **Input**: Original Question
            ```
            "Can you tell me about the regulations for stairs and handrails?"
            ```

            **Output**: Structured Query
            ```
            {
                "query": "fences, handrails",
                "filter": "or(contain(\"h3_subchapter\", \"Stairs\"), contain(\"h3_subchapter\", \"Hand rails\"))"
            }
            ```
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


        if selected_query_constructor == 'Self-Query-Construction':
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
        top_k = st.slider("Select the top k value for retrieval", 1, 20, 3, 1)
        st.session_state.top_k = top_k

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

        top_k = st.slider("Select the top k value for retrieval", 1, 20, 3, 1)
        st.session_state.top_k = top_k


    def self_query_retriever():
        st.markdown(
            '''
            ### Explanation 
            
            The Self-Query Retriever method uses structured queries, crafted from natural language by the Self-Query Constructor, to retrieve relevant documents from a VectorStore. This approach allows for precise metadata-based filtering along with semantic search to provide highly relevant document results.

            ### Example
            
            **Input**: Structured Query
            ```
            {
                "query": "highly rated science fiction film",
                "filter": "and(gt('rating', 8.5), eq('genre', 'science fiction'))"
            }
            ```

            **Output**: Retrieved Documents (top k = 1)
            ```
            [
                "A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
                "Three men walk into the Zone, three men walk out of the Zone"
            ]
            ```
            The output showcases documents that match the criteria set by the structured query's filters for genre and rating.
            '''
        )

    def self_query_retriever_settings():
        pass

    def multi_vector_retriever():
        st.markdown(
            '''
            ### Explanation
            
            The Multi-Vector Retriever efficiently handles complex queries by first retrieving document summaries and then the complete documents themselves. It works by comparing the semantic content of a user's natural language question to multiple vectors representing different aspects of documents within a vector database.

            ### Example
            
            **Input**: Original Question
            ```
            "Explain the impact of AI on society"
            ```

            **Intermediate Step**: Retrieved Summary from VectorStore
            ```
            "A summary document highlighting the key effects of artificial intelligence on modern social dynamics, workforce changes, and ethical debates."
            ```

            **Output**: Retrieved Full Document from InMemoryStore
            ```
            "A detailed analysis exploring artificial intelligence's transformative role in society, its potential to disrupt job markets, influence social interactions, and raise ethical issues. This comprehensive document delves into both the positive advancements AI brings, such as increased efficiency and new opportunities, as well as the challenges, including job displacement and privacy concerns."
            ```
            
            This process starts with a user's question and employs a two-step retrieval, first presenting a summarised snippet for quick understanding, followed by the full document for an in-depth exploration.
            '''
        )

    def multi_vector_retriever_settings():
        inmemorystore_folder = os.path.join(os.getcwd(), "inmemorystore/indexed_documents")
        jsonl_files = [f for f in os.listdir(inmemorystore_folder) if f.endswith('.jsonl')]
        if jsonl_files:
            options = ['None'] + jsonl_files
            selected_file = st.selectbox("Select a JSONL file:", options)

            if selected_file != 'None' and st.button("Load Selected File"):
                file_path = os.path.join(inmemorystore_folder, selected_file)
                st.session_state.mvr_documents_store = VectorSearch.load_documents_from_jsonl(file_path)
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
        st.session_state.top_filter_n = st.slider("Select the number of top results to filter", min_value=1, max_value=20, value=5, step=1) 

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
        with st.form(key="custom_prompt"):
            user_defined_prompt = st.text_area("Enter your custom prompt template here:", value=st.session_state.get('user_custom_prompt', ''))
            st.session_state.user_custom_prompt = user_defined_prompt  # Update session state

            full_custom_prompt = user_defined_prompt + "\n\nContext: {context}\n\nQuestion: {question}"
            st.session_state.full_custom_prompt = full_custom_prompt

            if st.form_submit_button("Update Example"):
                st.rerun()
        

class Explain_DL:

    @staticmethod
    def document_loader_langchain():
        st.markdown(
        '''
        ### Explanation

        The Document Loader LangChain method loads and processes documents, supporting various formats like markdown and PDF. It is designed to handle different types of documents and extract their content effectively.

        ### Functionality

        - **Format Support**: This loader can process markdown and PDF files, extracting text content from each format.
        - **Metadata Addition**: For each document, metadata such as file name and file type are added, providing contextual information about the document.
        - **Versatile Loading**: The method is adaptable, allowing for efficient handling of various document types, making it a flexible choice for diverse document processing needs.
        '''
    )

    def document_loader_langchain_settings():

        if st.checkbox("Enable Source Column Index Selection", value=False):
            column_index = st.number_input("Enter the header column index for source", min_value=0)
            st.session_state.source_column_index = column_index
        else:
            st.session_state.source_column_index = None

        if st.checkbox("Select Metadata Columns", key="select_metadata_columns"):
            if 'metadata_column_indexes' not in st.session_state:
                st.session_state.metadata_column_indexes = []

            # UI for adding metadata column indexes
            new_col_index = st.number_input("Metadata Column Index", min_value=0, key="new_metadata_col_index")
            if st.button("Add Metadata Column Index"):
                st.session_state.metadata_column_indexes.append(new_col_index)

            st.write("Current Metadata Column Indexes:", st.session_state.metadata_column_indexes)

            # Reset the list if needed
            if st.button("Reset Metadata Column Indexes"):
                st.session_state.metadata_column_indexes = []
                st.rerun()

    def document_loader_unstructured():
        pass

    def document_loader_unstructured_settings():
        pass

class Explain_DS:
    
    @staticmethod
    def recursive_splitter():
        st.markdown(
        '''
        ### Explanation

        The Recursive Splitter method divides text into chunks using a hierarchy of separators. It aims to maintain semantically related text together as long as possible. The method is parameterized by a list of characters and attempts to split the text on them in order, prioritizing larger semantic units.

        ### Functionality

        - **Text Splitting Logic**: The text is split based on a list of characters, such as `["\\n\\n", "\\n", " ", ""]`. The splitter tries these separators in order, starting from the largest (like double newlines indicating paragraphs) to the smallest (like individual spaces or empty strings).
        - **Chunk Size Measurement**: The size of each chunk is determined by the number of characters. The goal is to keep the chunk size within a specified limit while retaining the semantic integrity of the text as much as possible.
        - **Maintaining Semantic Units**: By using separators like newlines and spaces, the splitter tries to keep paragraphs, sentences, or words intact, ensuring that each chunk is as meaningful as possible.
        '''
        )

    def recursive_splitter_settings():
        chunk_size = st.slider(
            "Choose the size of text chunks",
            min_value=100, max_value=10000, value=3000, step=100
        )
        st.session_state.chunk_size = chunk_size
        
        chunk_overlap = st.slider(
            "Choose the overlap between text chunks",
            min_value=0, max_value=500, value=200, step=10
        )
        st.session_state.chunk_overlap = chunk_overlap

        separator_display_mapping = {
            "\\n\\n (Double Newline)": "\n\n",
            "\\n (Newline)": "\n",
            "Space": " ",
            "Empty String": ""
        }

        # At the start of your Streamlit app, initialize selected_separators
        if 'selected_separators' not in st.session_state:
            st.session_state.selected_separators = list(separator_display_mapping.values())

        # Then, in your UI logic, update selected_separators based on user interaction
        separator_options = list(separator_display_mapping.keys()) + ["ALL"]
        default_selections = list(separator_display_mapping.keys())  # Default to all separators selected

        selected_separators_display = st.multiselect(
            "Choose the separators for splitting",
            options=separator_options,
            default=default_selections
        )

        # If "ALL" is selected, use all separators, otherwise only selected ones
        if "ALL" in selected_separators_display:
            st.session_state.selected_separators = list(separator_display_mapping.values())
        else:
            st.session_state.selected_separators = [
                separator_display_mapping[disp] 
                for disp in selected_separators_display 
                if disp in separator_display_mapping
            ]

    def character_splitter():
        st.markdown(
        '''
        ### Explanation

        The Character Splitter method divides text into chunks based on a specific character, typically focusing on preserving the natural structure of the text like paragraphs or sentences. This method is straightforward and highly effective for texts where a certain character reliably indicates a logical separation.

        ### Functionality

        - **Character-based Splitting**: The primary logic of this method is to split text based on a single character, such as `"\n\n"` for double newlines. This approach aims to segment the text at meaningful points, like between paragraphs.
        - **Chunk Size Measurement**: The size of each chunk is measured in terms of the number of characters. The method ensures that each chunk does not exceed a predefined character limit, maintaining a uniform size distribution.
        - **Chunk Overlap**: To ensure continuity and context, there is an option to define an overlap between chunks. This overlap means that the end of one chunk and the beginning of the next will share some common text.

        The simplicity of this method makes it versatile and easily applicable to various types of documents, especially where clear textual separators are present.
        '''
    )

    def character_splitter_settings():
        chunk_size = st.slider(
            "Choose the size of text chunks",
            min_value=100, max_value=10000, value=3000, step=100
        )
        st.session_state.chunk_size = chunk_size
        
        chunk_overlap = st.slider(
            "Choose the overlap between text chunks",
            min_value=0, max_value=500, value=200, step=10
        )
        st.session_state.chunk_overlap = chunk_overlap
    
    @staticmethod
    def markdown_header_splitter():
        st.markdown(
        '''
        ### Explanation

        The Markdown Header Splitter method focuses on dividing a markdown document into chunks based on specified headers. This approach ensures that each chunk is contextually coherent, maintaining the structure and thematic grouping inherent in the document.

        ### Functionality

        - **Header-based Chunking**: The splitter identifies specific markdown headers (like `#`, `##`, `###`) to segment the document. Text under each header is grouped together, forming distinct chunks.
        - **Custom Header Mapping**: Users can specify which headers to split on, allowing flexibility to target different levels of the document hierarchy. For example, splitting on `#` and `##` headers creates chunks at major and sub-section levels.
        - **Chunk Size and Overlap Control**: Besides header-based splitting, users can define the size of text chunks and the overlap between them, providing additional control over how the text is divided.
        '''
        )

    def markdown_header_splitter_settings():
        chunk_size = st.slider(
            "Choose the size of text chunks",
            min_value=100, max_value=10000, value=3000, step=100
        )
        st.session_state.chunk_size = chunk_size
        
        chunk_overlap = st.slider(
            "Choose the overlap between text chunks",
            min_value=0, max_value=500, value=200, step=10
        )
        st.session_state.chunk_overlap = chunk_overlap

        separator_display_mapping = {
            "\\n\\n (Double Newline)": "\n\n",
            "\\n (Newline)": "\n",
            "Space": " ",
            "Empty String": ""
        }

        # At the start of your Streamlit app, initialize selected_separators
        if 'selected_separators' not in st.session_state:
            st.session_state.selected_separators = list(separator_display_mapping.values())

        # Then, in your UI logic, update selected_separators based on user interaction
        separator_options = list(separator_display_mapping.keys()) + ["ALL"]
        default_selections = list(separator_display_mapping.keys())  # Default to all separators selected

        selected_separators_display = st.multiselect(
            "Choose the separators for splitting",
            options=separator_options,
            default=default_selections
        )

        # If "ALL" is selected, use all separators, otherwise only selected ones
        if "ALL" in selected_separators_display:
            st.session_state.selected_separators = list(separator_display_mapping.values())
        else:
            st.session_state.selected_separators = [
                separator_display_mapping[disp] 
                for disp in selected_separators_display 
                if disp in separator_display_mapping
            ]

        default_headers = [("##", "h1_main"), ("###", "h2_chapter"), ("####", "h3_subchapter")]

        if 'headers_to_split_on' not in st.session_state:
            st.session_state.headers_to_split_on = default_headers.copy()

        # Function to add a new header
        def add_header():
            next_header_index = len(st.session_state.headers_to_split_on) + 1
            next_header_sign = "#" * next_header_index
            next_header_name = f"Header {next_header_index}"
            st.session_state.headers_to_split_on.append((next_header_sign, next_header_name))

        # Function to remove the last header
        def remove_header():
            if st.session_state.headers_to_split_on:
                st.session_state.headers_to_split_on.pop()
    

        # Display current headers and allow editing
        for index, (header_sign, header_name) in enumerate(st.session_state.headers_to_split_on):
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    new_header_sign = st.text_input(f"Header sign {index+1}", value=header_sign)
                with cols[1]:
                    new_header_name = st.text_input(f"Header name {index+1}", value=header_name)
                if new_header_sign != header_sign or new_header_name != header_name:
                    st.session_state.headers_to_split_on[index] = (new_header_sign, new_header_name)


        # Buttons to add/remove headers
        col1, col2 = st.columns(2)
        with col1:
            st.button("Add header", on_click=add_header)
        with col2:
            st.button("Remove header", on_click=remove_header)

        # Show the headers as they will be used in the splitter
        st.write("Headers to split on:")
        #for header in st.session_state.headers_to_split_on:
            #st.write(f"{header[0]}: {header[1]}")
        st.write(st.session_state.headers_to_split_on)

class Explain_DP:
    @staticmethod
    def clean_chunks_content():
        st.markdown(
            '''
            ### Explanation

            The Clean Chunks Content method is designed to refine and standardize text data by applying a series of text cleaning functions. This method enhances readability and uniformity, ensuring that the processed text is more analyzable and consistent.

            ### Functionality

            - **Text Cleaning Functions**: This method uses a range of functions to clean text, including removing extra whitespace, standardizing dashes and bullets, trimming trailing punctuation, and handling ordered bullets.
            - **Application on Document Chunks**: Each cleaning function is applied to every chunk of the document, refining the text content for better processing in subsequent stages.
            - **Enhanced Readability and Analysis**: By cleaning the text, this method improves the readability and uniformity of the text, making it more suitable for analysis and embedding processes.
            '''
        )

    def clean_chunks_content_settings():
        pass

    @staticmethod
    def customize_document_metadata():
        st.markdown(
            '''
            ### Explanation

            Customize Document Metadata method allows users to modify the metadata of document chunks. This approach provides flexibility in managing document information, enhancing the overall organization and retrieval process.

            ### Functionality

            - **Metadata Customization**: Users can select metadata components to add or remove, tailoring the metadata to fit specific needs or preferences.
            - **Unique ID Addition**: The method includes an option to add unique identifiers to each document chunk, either as a UUID or a combination of file name and UUID, ensuring easy tracking and reference.
            - **Flexible Metadata Handling**: This approach offers a dynamic way to handle metadata, accommodating diverse document processing requirements and enhancing the utility of the document corpus.
            '''
        )

    def customize_document_metadata_settings():
        all_removable_metadata_keys = ['source', 'file_name', 'file_type']
        default_removed_metadata_keys = ['source']

        # User selects which metadata components to remove
        remove_metadata_keys_selection = st.multiselect(
            "Select metadata components to remove:",
            options=all_removable_metadata_keys,
            default=default_removed_metadata_keys
        )
        st.session_state.remove_metadata_keys = remove_metadata_keys_selection

        # All possible metadata components to add
        all_addable_metadata_keys = ['unique_id']
        default_added_metadata_keys = ['unique_id']
        
        # User selects which metadata components to add
        add_metadata_keys_selection = st.multiselect(
            "Select metadata components to add:",
            options=all_addable_metadata_keys,
            default=default_added_metadata_keys
        )
        st.session_state.add_metadata_keys = add_metadata_keys_selection

        if 'unique_id' in add_metadata_keys_selection:
            unique_id_options = ['uuid', 'file_name + uuid']
            selected_unique_id_type = st.radio(
                "Select the type of unique identifier to add:",
                options=unique_id_options,
                index=unique_id_options.index('file_name + uuid'),  # Default to 'file_name + uuid'
                horizontal=True
            )
            st.session_state.selected_unique_id_type = selected_unique_id_type

    @staticmethod
    def filter_short_documents():
        st.markdown(
            '''
            ### Explanation

            The Filter Short Documents method aims to improve the quality of the document set by removing overly brief chunks. This filtering ensures that the retained documents have substantial content, enhancing their relevance and informativeness.

            ### Functionality

            - **Length-based Filtering**: Documents with content less than a specified character length (e.g., 50 characters) are filtered out.
            - **Focus on Content-rich Documents**: By removing shorter documents, this method prioritizes content-rich documents, which are likely to provide more valuable information for users.
            - **Enhanced Dataset Quality**: The result is a cleaner, more focused dataset, where each document contributes meaningfully to the overall corpus.
            '''
        )

    def filter_short_documents_settings():
        pass

    @staticmethod
    def build_index_schema():
        st.markdown(
            '''
            ### Explanation

            Build Index Schema method constructs a structured schema for indexing documents based on their headers. This structure facilitates efficient organization and retrieval of documents by categorizing them according to their hierarchical levels.

            ### Functionality

            - **Header-based Indexing**: The schema is built by identifying unique headers in the documents and using them to categorize the content.
            - **Flexible Schema Creation**: Depending on the document set, headers can be automatically extracted and sorted, or predefined headers can be used to create the schema.
            - **Enhanced Document Organization**: This method ensures documents are organized in a way that aligns with their inherent structure, making it easier to navigate and retrieve specific sections or topics.
            '''
        )

    def build_index_schema_settings():
        pass

    @staticmethod
    def build_toc_from_documents():
        st.markdown(
            '''
            ### Explanation

            The Build TOC from Documents method creates a Table of Contents (TOC) for a collection of documents based on their headers. This TOC reflects the hierarchical structure of the documents, providing an organized overview of their contents.

            ### Functionality

            - **Dynamic TOC Creation**: The TOC is generated by traversing through the headers of each document and constructing a nested structure that represents the document organization.
            - **Customizable Header Levels**: Users can specify which headers to include in the TOC, allowing for flexible representation of the document's structure.
            - **Structured Overview**: The resulting TOC provides a clear, structured overview of the entire document set, facilitating easy navigation and understanding of the content landscape.
            '''
        )

    def build_toc_from_documents_settings():
        pass

class Explain_DI:
    @staticmethod
    def summary_indexing():
        st.markdown(
            '''
            ### Explanation

            Summary Indexing involves creating a summary for each document and using those summaries for embedding and retrieval. This approach aims to distill the core content of documents, enhancing the accuracy of retrieval.

            ### Functionality

            - **Summarization of Content**: Each document is summarized to capture its essence, focusing on key points and themes.
            - **Enhanced Retrieval**: By embedding summaries rather than entire documents, the retrieval process becomes more targeted and precise.
            - **Versatility in Embedding**: Summaries can be embedded along with or instead of the full document text, offering flexibility in how documents are represented and retrieved.
            '''
        )

    def summary_indexing_settings():
        custom_file_name = st.text_input("Enter a name for the output JSONL file (leave blank for auto-naming):")
        if not custom_file_name:
            # Generate a unique file name using a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_file_name = f"original_documents_{timestamp}.jsonl"
        st.session_state.original_document_file_name = custom_file_name

    @staticmethod
    def parent_document_indexing():
        st.markdown(
            '''
            ### Explanation

            Parent Document Indexing splits documents into smaller chunks and indexes them, while retaining a link to their parent document. This method balances the need for detailed embeddings with maintaining contextual information.

            ### Functionality

            - **Chunk Splitting and Indexing**: Documents are split into smaller, more manageable chunks, which are then indexed for efficient retrieval.
            - **Context Retention**: Each chunk retains a connection to its parent document, ensuring that the broader context is not lost.
            - **Optimized Retrieval Process**: This approach offers a nuanced balance, enabling precise retrieval of relevant sections while keeping the larger context accessible.
            '''
        )

    def parent_document_indexing_settings():
        custom_file_name = st.text_input("Enter a name for the output JSONL file (leave blank for auto-naming):")
        if not custom_file_name:
            # Generate a unique file name using a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_file_name = f"original_documents_{timestamp}.jsonl"
        st.session_state.original_document_file_name = custom_file_name

class Explain_DE:
    @staticmethod
    def openai_embedding_model():
        st.markdown(
            '''
            ### Explanation

            The OpenAI Embedding Model utilizes OpenAI's advanced language models to generate embeddings for documents. These embeddings capture the semantic nuances of the text, making them highly effective for retrieval and analysis.

            ### Functionality

            - **Advanced Language Model**: Leveraging OpenAI's state-of-the-art language models ensures rich and nuanced embeddings.
            - **Semantic Richness**: The embeddings generated capture a wide range of semantic information, enhancing the effectiveness of retrieval and analysis tasks.
            - **Broad Applicability**: This method is suitable for a variety of text types, making it a versatile choice for embedding needs.
            '''
        )

    def openai_embedding_model_settings():
        pass

class Explain_VD:
    @staticmethod
    def build_chroma_vectorstore():
        st.markdown(
            '''
            ### Explanation

            Building a Chroma Vectorstore involves creating a specialized database for storing document embeddings. This vectorstore facilitates efficient storage and retrieval of embedded documents.

            ### Functionality

            - **Efficient Storage**: Chroma Vectorstore stores embeddings in an optimized format, enabling quick access and retrieval.
            - **Scalability**: The system is designed to handle large volumes of documents, scaling effectively as the dataset grows.
            - **Seamless Integration**: The Chroma Vectorstore integrates smoothly with the document embedding process, creating a cohesive pipeline for document handling and retrieval.
            '''
        )

    def build_chroma_vectorstore_settings():
        custom_collection_name = st.text_input("Enter a custom collection name:", value=st.session_state.collection_name)
            
        if custom_collection_name:
            st.session_state.custom_collection_name = custom_collection_name
            st.success(f"Custom collection name set to: {custom_collection_name}")

    @staticmethod
    def build_redis_vectorstore():
        st.markdown(
            '''
            ### Explanation

            Building a Redis Vectorstore entails setting up a Redis-based database for storing and managing document embeddings. This method leverages Redis's capabilities to handle vector data efficiently.

            ### Functionality

            - **Redis Database Utilization**: Redis is used for its high-performance data handling, particularly for vector data.
            - **Rapid Retrieval and Management**: The Redis Vectorstore offers fast retrieval times and efficient management of embedded documents.
            - **Customizable Configuration**: Users can configure the Redis connection and index names, tailoring the vectorstore to their specific needs.
            '''
        )

    def build_redis_vectorstore_settings():
        pass

