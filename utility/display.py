import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from typing import List
from langchain.schema import Document

class Retrieval_Stepper:

    def display_document_with_metadata(document, metadata):
        st.write('Page Content:', document.page_content)
        st.write('Metadata:')
        for key, value in metadata.items():
            st.write(f'{key.capitalize()}:', value)
    
    def display_query_construction_results(query_construction_results, container_key):
        if query_construction_results:
            with st.expander(f"Query Constructor Prompt ({container_key})"):
                st.markdown(query_construction_results.get('constructor_prompt', ''))

            with st.expander(f"Structured Request Output ({container_key})"):
                st.markdown(query_construction_results.get('structured_query', {}))
        else:
            st.write("No query construction results to display.")

    def display_vector_search_results(vector_search_results, container_key):
        # Setup pagination state
        current_doc_index_key = f'current_doc_index_{container_key}'
        if current_doc_index_key not in st.session_state:
            st.session_state[current_doc_index_key] = 0

        # Check if vector_search_results is a list of Document objects
        if vector_search_results and isinstance(vector_search_results, List) and all(isinstance(item, Document) for item in vector_search_results):
            current_doc_index = st.session_state[current_doc_index_key]
            #st.write(st.session_state[current_doc_index_key])
            num_documents = len(vector_search_results)
            st.write("Current Document Index:", current_doc_index)  # Debugging line
            st.write("Number of Documents:", len(vector_search_results))

            if current_doc_index >= num_documents:
                st.session_state[current_doc_index_key] = 0
                current_doc_index = 0

            if 0 <= current_doc_index < num_documents:
            # Create two columns for content and metadata
                col1, col2, col3 = st.columns([5, 2, 1])
                with col1:
                    st.subheader("Page Content:")
                    st.write(vector_search_results[current_doc_index].page_content)

                with col2:
                    st.subheader("Metadata:")
                    for key, value in vector_search_results[current_doc_index].metadata.items():
                        st.caption(f"{key}: {value}")

                with col3:
                    #st.subheader("Info")
                    # Display page number information
                    page_info = f"Page {current_doc_index + 1}/{num_documents}"
                    st.write(page_info)

                    if st.button("➡️", key=f"next_{container_key}"):
                        # Increment the current index and wrap around if necessary
                        st.session_state[current_doc_index_key] = (current_doc_index + 1) % num_documents
                        st.rerun()

                    # Previous button
                    if st.button("⬅️", key=f"previous_{container_key}"):
                        # Decrement the current index and wrap around if necessary
                        st.session_state[current_doc_index_key] = (current_doc_index - 1 + num_documents) % num_documents
                        st.rerun()
            else:
                st.error(f"Document index out of range: {current_doc_index}")
        elif vector_search_results:
            st.write(vector_search_results)
        else:
            st.warning("No search results to display.")

    def display_post_processing_results(vector_search_results, container_key, unique_identifier):
        # Setup pagination state
        current_doc_index_key = f'current_doc_index_{container_key}_{unique_identifier}'
        if current_doc_index_key not in st.session_state:
            st.session_state[current_doc_index_key] = 0

        # Check if vector_search_results is a list of Document objects
        if vector_search_results:
            current_doc_index = st.session_state[current_doc_index_key]
            #st.write(st.session_state[current_doc_index_key])
            num_documents = len(vector_search_results)

            if current_doc_index >= num_documents:
                st.session_state[current_doc_index_key] = 0
                current_doc_index = 0

            if 0 <= current_doc_index < num_documents:

            # Create two columns for content and metadata
                col1, col2, col3 = st.columns([5, 2, 1])
                with col1:
                    st.subheader("Page Content:")
                    st.write(vector_search_results[current_doc_index].page_content)

                with col2:
                    st.subheader("Metadata:")
                    for key, value in vector_search_results[current_doc_index].metadata.items():
                        st.caption(f"{key}: {value}")

                with col3:
                    next_button_key = f"next_{container_key}_{unique_identifier}"
                    prev_button_key = f"previous_{container_key}_{unique_identifier}"
                    #st.subheader("Info")
                    # Display page number information
                    page_info = f"Page {current_doc_index + 1}/{num_documents}"
                    st.write(page_info)

                    if st.button("➡️", key=next_button_key):
                        # Increment the current index and wrap around if necessary
                        st.session_state[current_doc_index_key] = (current_doc_index + 1) % num_documents
                        st.rerun()

                    # Previous button
                    if st.button("⬅️", key=prev_button_key):
                        # Decrement the current index and wrap around if necessary
                        st.session_state[current_doc_index_key] = (current_doc_index - 1 + num_documents) % num_documents
                        st.rerun()
            else:
                st.error(f"Document index out of range: {current_doc_index}")
        else:
            st.warning("No post-processing results to display.")

    def display_pipeline_step(pipeline_results, current_step, step_title, container_key):
        if current_step == 0:
            st.write(f"{step_title} Transformed Question:", pipeline_results.get('query_transformation'))

        elif current_step == 1:
            st.write(f"{step_title} Constructed Query:")
            query_construction_results = pipeline_results.get('query_construction')
            Retrieval_Stepper.display_query_construction_results(query_construction_results, container_key)

        elif current_step == 2:
            #st.subheader(f"Vector Search Results:")
            vector_search_results = pipeline_results.get('vector_search')
            Retrieval_Stepper.display_vector_search_results(vector_search_results, container_key)

        elif current_step == 3:
            st.write(f"{step_title} Post Processing Results:")
            post_processing_results = pipeline_results.get('post_processing')

            if post_processing_results and isinstance(post_processing_results, dict):
                for method_name, results in post_processing_results.items():
                    with st.expander(f"Results after {method_name}"):
                        unique_identifier = method_name  # or any other unique identifier for the step
                        Retrieval_Stepper.display_post_processing_results(results, container_key, unique_identifier)
            else:
                st.write("No post-processing results to display.")

        elif current_step == 4:
            st.write(f"{step_title} Prompting:", pipeline_results.get('prompting'))

        elif current_step == 5:
            st.write(f"{step_title} Answer:", pipeline_results.get('answer'))

    def display_containers(pipeline_results, current_step, step_title, key, container_key, config_info):
        with stylable_container(
            key=key,
            css_styles=f"""
                {{
                    max-height: 400px;  /* Set the maximum height you want */
                    overflow-y: auto;   /* Add a scrollbar if content exceeds the container height */
                    padding: 1rem;      /* Add some padding */
                    border: 0px solid #ddd; /* Add a border for better visual separation */
                }}
            """,
            ):
            
            Retrieval_Stepper.display_pipeline_step(pipeline_results, current_step, step_title, container_key)
            with st.expander("Show Configuration"):
                if config_info:
                    # Display the configuration dictionary as JSON
                    st.json(config_info)
                else:
                    st.write("No specific configuration to display.")

class Ingestion_Stepper:
    def display_pipeline_step(ingestion_results, current_step, step_title, container_key):
        if current_step == 0:
            st.write(f"{step_title} Loaded Documents:", ingestion_results.get('document_loading'))

        elif current_step == 1:
            st.write(f"{step_title} Splitted Documents:", ingestion_results.get('document_splitting'))

        elif current_step == 2:
            st.write(f"{step_title} Processed Documents:")
            document_processing_results = ingestion_results.get('document_processing')
            if document_processing_results:
                for method_name, results in document_processing_results.items():
                    with st.expander(f"Results after {method_name}"):
                        st.write(results)
            else:
                st.write("No document processing results to display.")

        elif current_step == 3:
            st.write(f"{step_title} Indexed Documents:", ingestion_results.get('document_indexing'))

        elif current_step == 4:
            st.write(f"{step_title} Document Embedding:", ingestion_results.get('document_embedding'))

        elif current_step == 5:
            st.write(f"{step_title} Vector Database:", ingestion_results.get('vector_database'))
            
    def display_containers(ingestion_results, current_step, step_title, key, container_key, config_info):
        with stylable_container(
            key=key,
            css_styles=f"""
                {{
                    max-height: 800px; /* Set the maximum height you want */
                    overflow-y: auto; /* Add a scrollbar if content exceeds the container height */
                    padding: 1rem; /* Add some padding */
                    border: 0px solid #ddd; /* Add a border for better visual separation */
                }}
            """,
            ):
            
            Ingestion_Stepper.display_pipeline_step(ingestion_results, current_step, step_title, container_key)
            with st.expander("Show Configuration"):
                if config_info:
                    st.json(config_info)
                else:
                    st.write("No specific configuration to display.")
        


