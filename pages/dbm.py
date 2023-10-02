import streamlit as st
import chromadb
from chromadb.config import Settings
from collections import defaultdict
from utility.ingestion import ingest_documents_db, DBStore
from utility.client import ClientDB


# Initialize ChromaDB Client

def get_existing_collections(client):
    collections = client.list_collections()
    existing_collectionss = sorted(collections, key=lambda x: x.name)
    return existing_collectionss

def on_selectbox_change():
    st.session_state.show_info = True



def main():
    st.set_page_config(page_title="DB MANAGEMENT", page_icon="🗃️", layout="wide")
    st.title("Database Management 🗃️")

    with st.empty():
        if 'delete' not in st.session_state:
            st.session_state['delete'] = False
        if 'client_db' not in st.session_state:
            st.session_state.client_db = ClientDB(collection_name=None, load_vector_store=False)

    

    existing_collections = st.session_state.client_db.get_existing_collections()
    selected_collection = st.selectbox('Select a collection:', existing_collections)

    collection = None
    if selected_collection:
        collection = st.session_state.client_db.client.get_collection(selected_collection)

    with st.sidebar:
        st.sidebar.title("Upload to Document Database")
        uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)  # You can specify the types of files you want to accept
        if uploaded_files:
            file_details = {"FileName": ["All Documents"], "FileType": [], "FileSize": []}

            # Populate file_details using traditional loops
            for file in uploaded_files:
                file_details["FileName"].append(file.name)
                file_details["FileType"].append(file.type)
                file_details["FileSize"].append(file.size)

            # Use selectbox to choose a file
            selected_file_name = st.sidebar.selectbox('Choose a file:', file_details["FileName"], on_change=on_selectbox_change)

            # Get the index of the file selected
            if selected_file_name != "All Documents":
                file_index = file_details["FileName"].index(selected_file_name) - 1
                st.sidebar.write("You selected:")
                st.sidebar.write("FileName : ", file_details["FileName"][file_index + 1])
                st.sidebar.write("FileType : ", file_details["FileType"][file_index])
                st.sidebar.write("FileSize : ", file_details["FileSize"][file_index])
            else:
                st.sidebar.write("You selected all uploaded documents.")

            # Add a note to remind the user to press the "Process" button
            if st.session_state.show_info:
                st.sidebar.info("**Note:** Remember to press the 'Process' button for the current selection.")
                st.session_state.show_info = False

            if st.sidebar.button("Process"):
                with st.spinner("Processing"):
                    if selected_file_name == "All Documents":
                        for file in uploaded_files:
                            ingest_documents_db(file, file.name, selected_collection)
                            st.success(f"{file.name} uploaded successfully!")
                    else:
                        selected_file = uploaded_files[file_index]
                        ingest_documents_db(selected_file, selected_file.name, selected_collection)
                        st.success("PDF uploaded successfully!")

    collection_tab, settings_tab = st.tabs(["Collection", "Settings"])

    with collection_tab:

        if collection:
            document_count = collection.count()
            st.subheader(f"There are {document_count} documents in the collection.")
            
            if document_count > 0:
                # Listing the first 10 documents
                documents = collection.peek(limit=document_count)

                # Displaying each document chunk with actions
                ids = documents['ids']
                contents = documents.get('documents', [])
                metadatas = documents.get('metadatas', [])

                parent_docs_dict = defaultdict(list)
                for doc_id, doc in zip(documents['ids'], documents['documents']):
                    # Extracting the file name from the doc_id
                    file_name = doc_id.rsplit('_', 1)[0]  # Split by the last underscore
                    parent_docs_dict[file_name].append(doc)
                
                # Create UI Elements to Select Parent Document
                parent_docs_sorted = sorted(parent_docs_dict.keys())
                parent_doc = st.radio('Select a parent document:', parent_docs_sorted)

                filtered_ids = [doc_id for doc_id in ids if doc_id.startswith(parent_doc)]
                #st.write(filtered_ids)

                if st.button(f"Delete Document {parent_doc}"):
                    st.session_state['delete'] = True

                if st.session_state.get('delete', False):
                    st.warning(f"Are you sure you want to delete all chunks in {parent_doc}? This action cannot be undone.")

                    if st.button("Yes, Delete All"):
                        st.write("Deleting the following IDs: ", filtered_ids)  # Displaying IDs being deleted
                        collection.delete(ids=filtered_ids)
                        st.session_state['deleted'] = True
            
                        # Reset 'delete' state to False
                        st.session_state['delete'] = False

                        st.experimental_rerun()

                if st.session_state.get('deleted', False):

                    st.success(f"The selected document deleted successfully!")
                    
                    st.session_state['deleted'] = False



                # Creating columns
                col1, col2 = st.columns(2)
                
                # First Column

                with col1:
                    st.subheader("Available IDs")
                    selected_chunk_id = st.selectbox('Select a chunk:', filtered_ids)
                    with st.expander("View All IDs"):
                        for idx, value in enumerate(filtered_ids, start=1):  # start=1 will start the index from 1 instead of 0
                            st.write(f"ID {idx}: {value}")
                    
                # Find the selected document
                idx = ids.index(selected_chunk_id)  # Find the index of selected_chunk_id in ids list
                selected_content = contents[idx] if idx < len(contents) else ""
                selected_metadata = metadatas[idx] if idx < len(metadatas) else ""
                
                # Display content
                with col2:
                    st.subheader("Content:")
                    st.write(selected_content)

                    # Display metadata and content for the selected chunk
                    st.subheader("Metadata")
                    if isinstance(selected_metadata, dict):
                        for key, value in selected_metadata.items():
                            st.write(f"{key}: {value}")
                    else:
                        st.write(selected_metadata)
                    
                    
                    if st.button(f"Delete Chunk"):
                        if st.confirm(f"Are you sure you want to delete {selected_chunk_id}?"):
                            collection.delete(selected_chunk_id)
                            st.success(f"{selected_chunk_id} deleted successfully!")
        else:
            st.error("Could not load the collection. Please check the collection name and try again.")

    with settings_tab:
        st.subheader("Settings 🛠️")

        with st.expander("List of Collections"):
            st.write(existing_collections)
        
        # Creating New Collection
        with st.expander("Create New Collection"):
            new_collection_name = st.text_input("Enter new collection name:")
            if st.button("Create Collection"):
                if new_collection_name:
                    try:
                        st.session_state.client_db.client.create_collection(new_collection_name)
                        st.session_state.create_collection_message = f"Collection {new_collection_name} created successfully!"
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error creating collection: {e}")
            
            if 'create_collection_message' in st.session_state:
                st.success(st.session_state.create_collection_message)
                # Clear the message from the session state after displaying it
                del st.session_state.create_collection_message

        # Deleting a Collection
        with st.expander("Delete a Collection"):
            selected_collection_to_delete = st.selectbox('Select a collection to delete:', existing_collections)
            if st.button("Delete Collection"):
                if selected_collection_to_delete:
                    try:
                        st.session_state.client_db.client.delete_collection(selected_collection_to_delete)
                        st.session_state.delete_collection_message = f"Collection {selected_collection_to_delete} deleted successfully!"
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting collection: {e}")
    
            if 'delete_collection_message' in st.session_state:
                st.success(st.session_state.delete_collection_message)
                # Clear the message from the session state after displaying it
                del st.session_state.delete_collection_message

        with st.expander("Rename a Collection"):
            selected_collection_to_rename = st.selectbox('Select a collection to rename:', existing_collections, key='rename')
            new_name = st.text_input("Enter new name:")
            if st.button("Rename Collection"):
                if selected_collection_to_rename and new_name:
                    try:
                        collection = st.session_state.client_db.client.get_collection(selected_collection_to_rename)
                        collection.modify(name=new_name)
                        st.session_state.rename_collection_message = f"Collection {selected_collection_to_rename} renamed to {new_name} successfully!"
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error renaming collection: {e}")
            
            if 'rename_collection_message' in st.session_state:
                st.success(st.session_state.rename_collection_message)
                # Clear the message from the session state after displaying it
                del st.session_state.rename_collection_message


if __name__ == "__main__":
    main()