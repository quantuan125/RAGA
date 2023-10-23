import os
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from st_pages import add_page_title
import langchain
import langchain
import pinecone
from UI.customstoggle import customstoggle
from UI.css import apply_css
from streamlit_extras.colored_header import colored_header
from utility.client import ClientDB
from agent.miracle import MRKL
from agent.tools import SummarizationTool
from UI.sidebar import sidebar
from collections import defaultdict
from langchain.schema import Document
from utility.sessionstate import Init
langchain.debug = True
langchain.verbose = True

@st.cache_data
def display_pdfs(s3_url):
    pdf_display = f'<embed src="{s3_url}" width="100%" height="800" type="application/pdf"></embed>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def on_selectbox_change():
    st.session_state.show_info = True

def reset_chat():
    st.session_state.messages = [{"roles": "assistant", "content": "Hi, I am Miracle. How can I help you?"}]
    st.session_state.history = []
    st.session_state.search_keywords = []
    st.session_state.doc_sources = []
    st.session_state.summary = None
    st.session_state.agent.clear_conversation()
    st.session_state.primed_document_response = None

def display_messages(messages):
    # Display all messages
    for msg in messages:
        st.chat_message(msg["roles"]).write(msg["content"])

def update_focused_mode():
    st.session_state.focused_mode = not st.session_state.get('focused_mode', False)
    if not st.session_state.focused_mode:
        st.session_state.pdf_display = False

def update_pdf_display():
    st.session_state.pdf_display = not st.session_state.get('pdf_display', False)

def main():

        load_dotenv()
        st.set_page_config(page_title="MIRACLE AGENT", page_icon="🦜️", layout="wide")
        apply_css()
        st.title("MIRACLE AGENT 🦜️")
        
        with st.empty():
            Init.initialize_session_state()
            Init.initialize_agent_state()
            Init.initialize_clientdb_state()
            Init.initialize_pinecone_state()

        with st.expander("READ ME BEFORE USING! 📘", expanded=False):
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
        
        with st.sidebar:

            existing_collections = st.session_state.client_db.get_existing_collections()
            existing_collections = [None] + existing_collections
            #st.write(existing_collections)
            if not existing_collections:
                st.warning("No collections available.")
            else:  # Check if there are any existing collections
                # Set the default index for the selectbox
                default_index = 0
                if st.session_state.selected_collection_state in existing_collections:
                    default_index = existing_collections.index(st.session_state.selected_collection_state)

                # Callback to handle changes in collection selection
                def on_change_selected_collection():
                    st.session_state.selected_collection_state = st.session_state.new_collection_state
                    st.session_state.s3_object_url = None
                    if st.session_state.new_collection_state is not None:
                        st.session_state.client_db = ClientDB(username=st.session_state.username, collection_name=st.session_state.selected_collection_state)
                        st.session_state.agent = MRKL()

                selected_collection = st.selectbox(
                    'Select a collection:',
                    existing_collections,
                    index=default_index,
                    key='new_collection_state',
                    on_change=on_change_selected_collection
                )

                # Update session state
                st.session_state.selected_collection_state = selected_collection

                # Debugging information (remove later)
                #st.write(f"Selected collection from session state: {st.session_state.selected_collection_state}")
                #st.write(f"Selected collection from selectbox: {selected_collection}")

            collection = None
            if selected_collection:
                collection = st.session_state.client_db.client.get_collection(selected_collection)
                #st.write(collection)
            
            focused_mode = st.checkbox(
                "Enable Focused Mode",
                value=st.session_state.get('focused_mode', False),
                key="focused_mode_key",
                on_change=update_focused_mode
            )

            if focused_mode:
                if collection:
                    document_count = collection.count()
                    if document_count == 0:
                        st.warning("This collection has no documents.")

                    if document_count > 0:
                        documents = collection.peek(limit=document_count)
                        parent_docs_dict = defaultdict(list)
                        for doc_id, doc in zip(documents['ids'], documents['documents']):
                            file_name = doc_id.rsplit('_', 1)[0]  # Split by the last underscore
                            parent_docs_dict[file_name].append(doc)

                        parent_docs_sorted = sorted(parent_docs_dict.keys())
                        
                        # Dropdown for selecting a specific document
                        selected_document = st.selectbox("Select a Document:", [None] + parent_docs_sorted)
                        st.session_state.selected_document = selected_document
                         

                        pdf_display = st.checkbox(
                            "Enable PDF Display",
                            value=st.session_state.get('pdf_display', False),
                            key="pdf_display_key",
                            on_change=update_pdf_display
                        )
                        
                        if selected_document is None:
                            st.info("No document is selected.")
                            st.session_state.s3_object_url = None

                        else:
                            st.session_state.agent = MRKL()
                            document_data = collection.get(where={"file_name": {"$eq": selected_document}}, include=["documents", "metadatas"])
                            
                            #st.write(f"Debug: document_data['metadatas'] = {document_data['metadatas']}")

                            #st.write(f"Debug: document_data['documents'] = {document_data['documents']}")

                            selected_document_chunks = document_data["documents"]
                            #st.write(selected_document_chunks)

                            # Additional metadata extraction and other operations can go here.
                            first_document_metadata = document_data.get("metadatas", [{}])[0]

                            # Extract the S3 file URL from the metadata
                            file_url = first_document_metadata.get("file_url")

                            # Update the session state to hold the S3 file URL
                            st.session_state.s3_object_url = file_url

                            # Display a message indicating that a document has been selected
                            st.write(f"You have selected '{selected_document}'")

                            if st.button("Create Detailed Summary"):
                                with st.spinner("Summarizing"):
                                    document_objects = [Document(page_content=chunk) for chunk in selected_document_chunks]
                                    summarization_tool = SummarizationTool(document_chunks=document_objects)
                                    st.session_state.summary = summarization_tool.run()
                                    # Display the summary
                                    st.session_state.messages.append({"roles": "assistant", "content": st.session_state.summary})



            sidebar.file_upload_and_ingest(st.session_state.client_db, selected_collection, collection, on_selectbox_change)


        main_chat_tab, chat_setting_tab = st.tabs(["Main Chat", "Chat Settings"])

        with main_chat_tab:
            if st.session_state.get('focused_mode', False) and st.session_state.get('pdf_display', False):
                # Two columns: one for the PDF display, and one for the chat
                col1, col2 = st.columns([1, 1])

                # PDF Display in Column 1
                with col1:
                    if st.session_state.s3_object_url is not None:  # Assuming you've set this session state variable
                        s3_url = st.session_state.s3_object_url
                        display_pdfs(s3_url)
                    else:
                        st.warning("Select a Document in the sidebar to display in this chat")

                # Main Chat in Column 2
                with col2:
                    display_messages(st.session_state.messages)

            else:
                # Only display the chat, since PDF display is not enabled
                display_messages(st.session_state.messages)

        with chat_setting_tab:
            st.info("Feature in development")
                

        if user_input := st.chat_input("Type something here..."):
            st.session_state.user_input = user_input
            st.session_state.messages.append({"roles": "user", "content": st.session_state.user_input})
            st.chat_message("user").write(st.session_state.user_input)

            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True, collapse_completed_thoughts = False, max_thought_containers = 1)
                result = st.session_state.agent.run_agent(input=st.session_state.user_input, callbacks=[st_callback])
                st.session_state.result = result
                response = result.get('output', '')
                st.session_state.messages.append({"roles": "assistant", "content": response})
                st.experimental_rerun()


        #with st.expander("Cost Tracking", expanded=True):
            #total_token = st.session_state.token_count
            #st.write(total_token)

        st.divider()
        buttons_placeholder = st.container()
        with buttons_placeholder:
            #st.button("Regenerate Response", key="regenerate", on_click=st.session_state.agent.regenerate_response)
            st.button("Clear Chat", key="clear", on_click=reset_chat)

            relevant_keys = ["Header ", "Header 3", "Header 4", "page_number", "source", "file_name", "title", "author", "snippet", "unique_id"]
            if st.session_state.doc_sources:
                content = []
                for document in st.session_state.doc_sources:
                    doc_dict = {
                        "page_content": document.page_content,
                        "metadata": {key: document.metadata.get(key, 'N/A') for key in relevant_keys}
                        }
                    content.append(doc_dict)
                
                customstoggle(
                    "Source Documents/Searched Links",
                    content,
                    metadata_keys=relevant_keys
                )

        if st.session_state.summary is not None:
            with st.expander("Show Summary"):
                st.subheader("Summarization")
                result_summary = st.session_state.summary
                st.write(result_summary)

        #st.write(st.session_state.history)
        #st.write(st.session_state.messages)
        st.write(st.session_state.br18_vectorstore)
        #st.write(st.session_state.br18_appendix_child_vectorstore)
        st.write(st.session_state.vector_store)
        #st.write(st.session_state.client_db)
        #st.write(st.session_state.agent)
        #st.write(st.session_state.result)
        #st.write(st.session_state.selected_collection)
        #st.write(st.session_state.s3_object_url)
        #st.write("Enable BR18", st.session_state.br18_exp)
        #st.write("Search Type", st.session_state.search_type)
        #st.write("Enable Websearch", st.session_state.web_search)
        #st.write("N Search Results", st.session_state.websearch_results)
        #st.write(st.session_state.llm_model)
        #st.write(st.session_state.streaming)
        #st.write(st.session_state.max_token_limit)



if __name__== '__main__':
    main()




    
