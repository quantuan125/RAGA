import streamlit as st
from utility.client import ClientDB
from utility.s3 import S3
from agent.miracle import MRKL
from streamlit_extras.colored_header import colored_header



class Main:

    @staticmethod
    def handle_collection_selection(existing_collections):
        def on_change_selected_collection():
            st.session_state.selected_collection_state = st.session_state.new_collection_state
            st.session_state.s3_object_url = None 
            if st.session_state.new_collection_state is not None:
                actual_collection_name = f"{st.session_state.username}-{st.session_state.new_collection_state}"
                st.session_state.client_db = ClientDB(username=st.session_state.username, collection_name=actual_collection_name)
                st.session_state.agent = MRKL()

        display_collections = [col.split('-', 1)[1] for col in existing_collections if col is not None]

        # Insert a "None" option at the beginning of the display list
        display_collections.insert(0, "None")

        default_index = 0
        if st.session_state.get('selected_collection_state'):
            actual_collection_name = f"{st.session_state.username}-{st.session_state.selected_collection_state}"
            if actual_collection_name in existing_collections:
                default_index = existing_collections.index(actual_collection_name) + 1  # Adjusted for "None" at index 0

        selected_collection = st.selectbox(
            'Select a collection:',
            display_collections,
            index=default_index,
            key='new_collection_state',
            on_change=on_change_selected_collection
        )

        collection_object = None
        actual_collection_name = None
        if selected_collection and selected_collection != "None":
            actual_collection_name = f"{st.session_state.username}-{selected_collection}"
            collection_object = st.session_state.client_db.client.get_collection(actual_collection_name)

        return actual_collection_name, collection_object
    
    @staticmethod
    def get_display_collections(existing_collections):
        display_collections = [name.split('-', 1)[1] if '-' in name else name for name in existing_collections]
        return display_collections
    
    @staticmethod
    def create_new_collection():
        new_collection_name_input = st.text_input("Enter new collection name:")
        if st.button("Create Collection"):
            if new_collection_name_input:
                new_collection_name = f"{st.session_state.username}-{new_collection_name_input}"  # Prefix the username
                try:
                    st.session_state.client_db.client.create_collection(new_collection_name)
                    st.session_state.create_collection_message = f"Collection {new_collection_name_input} created successfully!"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error creating collection: {e}")

        if 'create_collection_message' in st.session_state:
            st.success(st.session_state.create_collection_message)
            # Clear the message from the session state after displaying it
            del st.session_state.create_collection_message

    @staticmethod
    def delete_collection(existing_collections):
        display_collections = Main.get_display_collections(existing_collections)
        delete_collection_selection = st.selectbox('Select a collection to delete:', display_collections, key='delete_collection')
        delete_collection = f"{st.session_state.username}-{delete_collection_selection}"

        if st.button("Delete Collection"):
            if delete_collection:
                try:
                    st.session_state.client_db.client.delete_collection(delete_collection) 

                    s3_instance = S3()
                    s3_instance.delete_objects_in_collection(st.session_state.username, delete_collection)

                    st.session_state.delete_collection_message = f"Collection {delete_collection_selection} deleted successfully!"

                    st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error deleting collection: {e}")

        if 'delete_collection_message' in st.session_state:
            st.success(st.session_state.delete_collection_message)
            # Clear the message from the session state after displaying it
            del st.session_state.delete_collection_message

    @staticmethod
    def rename_collection(existing_collections):   
        display_collections = Main.get_display_collections(existing_collections)

        rename_collection_selection = st.selectbox('Select a collection to rename:', display_collections, key='rename_collection')
        rename_collection = f"{st.session_state.username}-{rename_collection_selection}"
        #st.write(rename_collection)

        new_name_input = st.text_input("Enter new name:")
        new_name = f"{st.session_state.username}-{new_name_input}"
        #st.write(new_name)

        if st.button("Rename Collection"):
            if rename_collection and new_name_input:
                try:
                    collection = st.session_state.client_db.client.get_collection(rename_collection)

                    collection.modify(name=new_name)

                    # Updating the file_url metadata
                    documents = collection.get()  # Get all documents in the collection
                    print(documents)
                    updated_metadatas = []  # List to hold updated metadata objects
                    ids = []  # List to hold document IDs

                    # Use documents['metadatas'] to access the list of metadata dictionaries
                    for document_metadata in documents['metadatas']:
                        old_url = document_metadata['file_url']

                        print(old_url)
                        # Split the old_url into its components
                        url_parts = old_url.split('/')
    
                        # Replace the collection name in the URL
                        url_parts[-2] = f"{st.session_state.username}-{new_name_input}"
                        
                        # Reassemble the URL
                        new_url = '/'.join(url_parts)

                        print(new_url)
                        
                        updated_metadatas.append({"file_url": new_url})
                        ids.append(document_metadata['unique_id'])

                    # Update the metadata in ChromaDB
                    collection.update(ids=ids, metadatas=updated_metadatas)

                    s3_instance = S3()  # Assuming you have a default constructor; adjust as necessary
                    s3_instance.rename_objects_in_collection(st.session_state.username, rename_collection, new_name)

                    st.session_state.rename_collection_message = f"Collection {rename_collection_selection} renamed to {new_name_input} successfully!"
                    st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error renaming collection: {e}")
                    print(e)
        
        if 'rename_collection_message' in st.session_state:
            st.success(st.session_state.rename_collection_message)
            # Clear the message from the session state after displaying it
            del st.session_state.rename_collection_message


class MainChat:
    @staticmethod
    def read_me_expander():
        st.markdown("""
        ## 🦜️ Welcome to Miracle! 
        Miracle is powered by **gpt-3.5-turbo**, specializing in construction, legal frameworks, and regulatory matters. 
        
        Below is a guide to help you navigate and understand the functionalities of this application better.
        """)
        
        colored_header(label="🛠️ Functionalities", color_name="blue-70", description="")
        
        st.markdown("""
        #### 1. **BR18 Feature** (Experimental)
        - **Enable BR18**: Integrate BR18 as part of Miracle's internal knowledge. You can toggle this feature in the sidebar.
        - **Search Types**:
            - **Header Search**: Searches by the headers in BR18. Recommend for specific queries
            - **Context Search**: Searches by content of paragraphs in BR18. Recommend for general queries

        #### 2. **Web Search Feature** (Experimental)
        - **Enable Web Search**: Integrate Google Search with up to 5 top results. You can adjust the number of results in the sidebar.

        #### 3. **Document Database**
        - **Upload & Process Document**: Upload PDFs as unstructured text and process them for Miracle to understand. Only one document can be processed at a time.
        - **Create Detailed Summary**: After processing a document, you can create a detailed summary of it. This might take 1-2 minutes.
        """)
        
        colored_header(label="📑 UI Interface", color_name="orange-70", description="")
        
        st.markdown("""
        #### 1. **Main Chat**: 
        - **View Source/Search Results**: Examine the results used by Miracle to produce its final answer.
        - **Clear Chat**: Resets the chat interface but does not reset functionalities.
                    
        #### 2. **PDF Display**: 
        - View your uploaded PDFs here. This tab only appears when a document is processed.
        """)
        
        colored_header(label="📜 SYSTEM PROMPT", color_name="yellow-70", description="")
        
        st.markdown("""
        For transparency, here is the initial prompt engineered for Miracle:

        ```
        You are Miracle, an expert in construction, legal frameworks, and regulatory matters.

        You have the following tools to answer user queries, but only use them if necessary. 

        Your primary objective is to provide responses that:
        1. Offer an overview of the topic, referencing the chapter and the section if relevant.
        2. List key points in bullet-points or numbered list format, referencing the clauses and their respective subclauses if relevant.
        3. Always match or exceed the details of the tool's output text in your answers. 
        4. Reflect back to the user's question and give a concise conclusion.
        5. If the search tool is used, you must always return the list of available URLs as part of your final answer. 

        Reminder: 
        Always try all your tools to find the answer to the user query

        Always self-reflect your answer based on the user's query and follows the list of response objective. 
        ```
        """)
        
        colored_header(label="🔗 Links", color_name="blue-green-70", description="")
        
        st.markdown("""
        - For any further assistance or more information, please contact <a href="mailto:qung@arkitema.com">qung@arkitema.com</a>.
        """, unsafe_allow_html=True)