import streamlit as st
import chromadb
from collections import defaultdict
from dotenv import load_dotenv
from UI.sidebar import sidebar
import boto3
import os
from utility.sessionstate import Init


# Initialize ChromaDB Client

def get_existing_collections(client):
    collections = client.list_collections()
    existing_collectionss = sorted(collections, key=lambda x: x.name)
    return existing_collectionss

def on_selectbox_change():
    st.session_state.show_info = True


def main():
    load_dotenv()
    st.set_page_config(page_title="DB MANAGEMENT", page_icon="🗃️", layout="wide")
    st.title("Database Management 🗃️")

    with st.empty():
        Init.initialize_session_state()
        Init.initialize_clientdb_state()
        

    existing_collections = st.session_state.client_db.get_existing_collections()
    existing_collections = [None] + existing_collections
    if not existing_collections:  # No existing collections
        st.warning("There are no existing collections. Please create a new collection to get started.")
        new_collection_name = st.text_input("Enter the name of the new collection:")
        if st.button("Create New Collection"):
            if new_collection_name:
                # Code to create a new collection
                st.session_state.client_db.client.create_collection(new_collection_name)
                st.success(f"Collection {new_collection_name} created successfully!")
                st.experimental_rerun()  # Rerun the script to refresh the state and UI
            else:
                st.error("Please enter a valid name for the new collection.")
    else:
        def on_change_selected_collection_dbm():
            st.session_state.selected_collection_state = st.session_state.new_collection_state_dbm
            if st.session_state.new_collection_state_dbm is not None:
                st.session_state.reinitialize_client_db = True

        # Set the default index for the selectbox
        default_index = 0
        if st.session_state.selected_collection_state in existing_collections:
            default_index = existing_collections.index(st.session_state.selected_collection_state)

        selected_collection = st.selectbox(
            'Select a collection:',
            existing_collections,
            index=default_index,
            key='new_collection_state_dbm',
            on_change=on_change_selected_collection_dbm
        )

        # Debugging information (remove later)
        #st.write(f"Selected collection from session state: {st.session_state.selected_collection_state}")
        #st.write(f"Selected collection from selectbox: {selected_collection}")

        collection = None
        if selected_collection:
            collection = st.session_state.client_db.client.get_collection(selected_collection)

        with st.sidebar:
            sidebar.file_upload_and_ingest(st.session_state.client_db, selected_collection, collection, on_selectbox_change)

        collection_tab, settings_tab = st.tabs(["Collection", "Settings"])

        with collection_tab:

            if collection:
                document_count = collection.count()
                #st.subheader(f"There are {document_count} pages in the collection.")
                
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
                    st.subheader(f"There are {len(parent_docs_sorted)} documents in the collection.")
                    parent_doc = st.radio('Select a document:', parent_docs_sorted)

                    filtered_ids = [doc_id for doc_id in ids if doc_id.startswith(parent_doc)]
                    #st.write(filtered_ids)

                    filtered_docs = [(doc_id, metadatas[ids.index(doc_id)]) for doc_id in filtered_ids]
                

                    if st.button(f"Delete '{parent_doc}'"):
                        st.session_state['delete'] = True

                    if st.session_state.get('delete', False):
                        st.warning(f"Are you sure you want to delete all chunks in {parent_doc}? This action cannot be undone.")

                        if st.button("Yes, Delete All"):
                            s3_urls_to_delete = [metadata.get('file_url') for metadata in metadatas if metadata.get('unique_id') in filtered_ids]
                            st.write(s3_urls_to_delete)
        
                            # Delete files from S3
                            if s3_urls_to_delete:
                                s3 = boto3.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION'),
                                                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

                                bucket_name = os.getenv("AWS_BUCKET_NAME")

                                for s3_url in s3_urls_to_delete:
                                    s3_key = s3_url.split(f"{bucket_name}/")[1]
                                    s3.delete_object(Bucket=bucket_name, Key=s3_key)

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
                        combined_ids = [f"(Page {doc_metadata['page_number']}): {doc_id}" for doc_id, doc_metadata in filtered_docs]
                        selected_chunk_id_combined = st.selectbox('Select a chunk:', combined_ids)
                        selected_chunk_id = selected_chunk_id_combined.split(": ", 1)[1]

                        with st.expander("View All IDs"):
                            for combined_id in combined_ids:
                                st.write(combined_id)
                        
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
                        
                        
                        if st.button(f"Delete Page"):
                            st.session_state['delete_chunk'] = True

                        if st.session_state.get('delete_chunk', False):
                            st.warning(f"Are you sure you want to delete {selected_chunk_id}? This action cannot be undone.")

                            if st.button("Yes, Delete"):
                                collection.delete(selected_chunk_id)
                                st.session_state['deleted_chunk'] = True
                                
                                # Reset 'delete_chunk' state to False
                                st.session_state['delete_chunk'] = False

                                st.experimental_rerun()

                        if st.session_state.get('deleted_chunk', False):
                            st.success(f"{selected_chunk_id} deleted successfully!")
                            
                            # Reset 'deleted_chunk' state
                            st.session_state['deleted_chunk'] = False
            else:
                st.error("Please select a collection name to continue.")

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

            with st.expander("Reset Client"):
                if st.button("Reset Client"):
                    st.session_state['reset'] = True
                
                if st.session_state.get('reset', False):
                    st.warning("Are you sure you want to reset the client? This action cannot be undone and will delete all collections and documents.")
                    
                    if st.button("Yes, Reset Client"):
                        # Reset 'reset' state to False immediately after confirmation
                        st.session_state['reset'] = False
                        
                        # Perform the client reset
                        st.session_state.client_db.reset_client()
                        
                        # Rerun the script to refresh the state and UI
                        st.experimental_rerun()

    

if __name__ == "__main__":
    main()