import streamlit as st
from utility.ingestion import DBStore

class Sidebar:

    @staticmethod
    def file_upload_and_ingest(client_db, collection_name, collection_object, on_selectbox_change):
        if st.session_state.get("upload_success", False):
            st.sidebar.success("PDF uploaded successfully!")
            st.session_state.upload_success = False

        st.sidebar.title("Upload Document to Database")
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

            if st.session_state.show_info:
                st.sidebar.info("**Note:** Remember to press the 'Process' button for the current selection.")
                st.session_state.show_info = False

            if st.sidebar.button("Process"):
                with st.spinner("Processing"):
                    if selected_file_name == "All Documents":
                        for file in uploaded_files:
                            db_store = DBStore(client_db, file_path=None, file_name=file.name, collection_name=collection_name)
                            if collection_object:  # This line is specific to dbm.py
                                existing_document_ids = db_store.check_document_exists(collection_object)
                                if existing_document_ids:
                                    collection_object.delete(ids=existing_document_ids)
                            db_store.ingest_document(file, collection_name)
                            st.session_state.upload_success = True
                    else:
                        selected_file = uploaded_files[file_index]
                        db_store = DBStore(client_db, file_path=None, file_name=selected_file.name, collection_name=collection_name)
                        if collection_object:  # This line is specific to dbm.py
                            existing_document_ids = db_store.check_document_exists(collection_object)
                            if existing_document_ids:
                                collection_object.delete(ids=existing_document_ids)
                        db_store.ingest_document(selected_file, collection_name)
                        st.session_state.upload_success = True
                    st.experimental_rerun()
            
            if st.session_state.get("upload_success", False):
                st.sidebar.success("PDF uploaded successfully!")
                st.session_state.upload_success = False 
                    
