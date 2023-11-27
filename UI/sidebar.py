import streamlit as st
from utility.ingestion import DBStore, DBIndexing
import time






class Sidebar:

    @staticmethod
    def file_upload_and_ingest(client_db, selected_collection_name, selected_collection_object):

        def on_selectbox_change():
                st.sidebar.info("**Note:** Remember to press the 'Process' button for the current selection.")
                st.session_state.show_info = False

        def process_all_documents(files, ingestion_class):
            for file in files:
                db_store = ingestion_class(client_db, file_name=file.name, collection_name=selected_collection_name)

                if selected_collection_object:
                    existing_document_ids = db_store.check_document_exists(selected_collection_object)
                    if existing_document_ids:
                        selected_collection_object.delete(ids=existing_document_ids)

                if ingestion_class == DBIndexing:
                    db_store.ingest_document(file, selected_collection_name, batch_size, delay)

                elif ingestion_class == DBStore:
                    db_store.ingest_document(file, selected_collection_name)
                
                st.session_state.upload_success = True
                

        def process_selected_document(file, ingestion_class):
            db_store = ingestion_class(client_db, file_name=file.name, collection_name=selected_collection_name)

            if selected_collection_object:
                existing_document_ids = db_store.check_document_exists(selected_collection_object)
                if existing_document_ids:
                    selected_collection_object.delete(ids=existing_document_ids)

            if ingestion_class == DBIndexing:
                db_store.ingest_document(file, selected_collection_name, batch_size, delay)

            elif ingestion_class == DBStore:
                db_store.ingest_document(file, selected_collection_name)
                
            st.session_state.upload_success = True

        st.sidebar.title("Upload Document to Database")
        new_ingestion = st.sidebar.checkbox("Smart Ingestion")
        Ingestion = DBIndexing if new_ingestion else DBStore
        #st.write(Ingestion)
        
        uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)
        if uploaded_files:
            file_details = {"FileName": ["All Documents"], "FileType": [], "FileSize": []}

            for file in uploaded_files:
                file_details["FileName"].append(file.name)
                file_details["FileType"].append(file.type)
                file_details["FileSize"].append(file.size)

            selected_file_name = st.sidebar.selectbox('Choose a file:', file_details["FileName"], on_change=on_selectbox_change)

            if selected_file_name != "All Documents":
                file_index = file_details["FileName"].index(selected_file_name) - 1
                st.sidebar.write("You selected:")
                st.sidebar.write("FileName : ", file_details["FileName"][file_index + 1])
                st.sidebar.write("FileType : ", file_details["FileType"][file_index])
                st.sidebar.write("FileSize : ", file_details["FileSize"][file_index])
            else:
                st.sidebar.write("You selected all uploaded documents.")

            if st.sidebar.button("Process"):
                batch_size = st.session_state.batch_size
                delay = st.session_state.delay

                if not selected_collection_name or selected_collection_name == "None":
                    st.sidebar.error("Please select a valid collection before processing.")
                else:
                    with st.spinner("Processing"):
                        if selected_file_name == "All Documents":
                            process_all_documents(uploaded_files, Ingestion)

                        else:
                            selected_file = uploaded_files[file_index]
                            process_selected_document(selected_file, Ingestion)
                        #st.experimental_rerun()
            
            if st.session_state.get("upload_success", False):
                st.sidebar.success("PDF uploaded successfully!")
                st.session_state.upload_success = False 
                    
