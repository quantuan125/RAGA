import streamlit as st
from dotenv import load_dotenv
from UI.ingestion_main import Ingestion_Settings, Ingestion_Explain, Ingestion_Display
from UI.retrieval_main import Retrieval_Settings, Retrieval_Explain, Retrieval_Display
from RAG.retrieval_pipeline import Retrieval_Pipeline
from UI.sidebar import UI_Sidebar
from UI.css import apply_css
from utility.sessionstate import Init
from langchain.vectorstores.redis import Redis
from langchain.embeddings import OpenAIEmbeddings
import os
import json
import langchain
from langchain.utilities.sql_database import SQLDatabase

#langchain.debug=True

def main():
    load_dotenv()
    st.set_page_config(page_title="Main", page_icon="", layout="wide")
    apply_css()
    st.title("MAIN 🔍")

    with st.empty():
        Init.initialize_session_state()
        Init.initialize_clientdb_state()
        if 'setting_last_selection' not in st.session_state:
            st.session_state.setting_last_selection = None
        if 'vector_search_selection' not in st.session_state:
            st.session_state.vector_search_selection = 'None'
        if 'expanded_setting' not in st.session_state:
            st.session_state.expanded_setting = None
        if 'baseline_pipeline_results' not in st.session_state:
            st.session_state.baseline_pipeline_results = {}
        if 'customized_pipeline_results' not in st.session_state:
            st.session_state.customized_pipeline_results = {}
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0


    with st.sidebar:
        #VECTORSTORE SELECTION
        with st.container():
            database_options = ['None', 'Chroma', 'Redis', 'SQLite', 'Postgres']

            selected_database = st.selectbox(
                "Select the vectorstore type",
                options=database_options,
                index=0,  # Default to None
            )

            if selected_database == 'None':
                st.session_state.vector_store = None
                st.warning("No vectorstore selected.")
                # Retrieval_Pipeline.check_and_initialize_vector_search()



            elif selected_database == 'Redis':
                redis_url = st.session_state.get('redis_url', 'redis://localhost:9000')
                redis_index_name = st.session_state.get('redis_index_name', 'user')


                # Load index schema
                index_schema_file_path = os.path.join("json/schema", "index_schema.json")
                with open(index_schema_file_path, 'r') as file:
                    redis_index_schema = json.load(file)

                st.session_state.redis_index_schema = redis_index_schema

                # Initialize Redis vectorstore
                st.session_state.vector_store = Redis(
                    redis_url=redis_url,
                    index_name=redis_index_name,
                    embedding=OpenAIEmbeddings(),
                    index_schema=redis_index_schema,
                )
                Retrieval_Pipeline.check_and_initialize_vector_search()

            elif selected_database == 'Chroma':

                existing_collections = st.session_state.client_db.get_existing_collections()
                if not existing_collections:
                    st.warning("No collections available.")
                else:
                    selected_collection_name, selected_collection_object = UI_Sidebar.handle_collection_selection(existing_collections)
                    st.session_state.collection_name = selected_collection_name
                    #VectorSearch(st.session_state.vector_store)
                    Retrieval_Pipeline.check_and_initialize_vector_search()

            elif selected_database == 'SQLite':
                db_path = "sqlite:///storage/SQL/Chinook.db"
                db = SQLDatabase.from_uri(db_path)
                st.session_state.database = db

            
            elif selected_database == 'Postgres':
                postgres_databases = ['chinook_pg_main', 'gln']

                # Let the user select a database
                selected_postgres_db = st.selectbox(
                    "Select a PostgreSQL database",
                    options=postgres_databases,
                    index=0,  # Default to the first database
                )

                # Fetch the base URI from environment variables
                db_uri_base = os.environ.get('POSTGRES_URI')

                # Construct the complete database path
                db_path = f"{db_uri_base}{selected_postgres_db}"

                # Initialize the database
                db = SQLDatabase.from_uri(db_path)
                st.session_state.database = db

            st.write(st.session_state.database)
            st.write(st.session_state.vector_store)
            
                #Sidebar.file_upload_and_ingest(st.session_state.client_db, selected_collection_name, selected_collection_object)
                #Pipeline.check_and_initialize_vector_search()

                # st.write("Current vector_store:", st.session_state.vector_store)
                # st.write("Current vector_search_instance:", Pipeline.vector_search_instance)
               
        #DISPLAY SELECTION
        with st.container():
            column_options = {
                'All Columns': 'all',
                'Settings and Explain': 'settings_explain',
                'Display': 'display'
            }

            # Default to showing all columns
            if 'selected_column_option' not in st.session_state:
                st.session_state.selected_column_option = 'Settings and Explain'


            # Display the radio buttons and store the current selection in session state
            selected_column_option = st.radio(
                "Select which columns to display",
                options=list(column_options.keys()),
                index=list(column_options.keys()).index(st.session_state.selected_column_option),
                key='selected_column_option'
            )
            
            st.session_state.show_baseline = st.sidebar.toggle("Show Baseline Results", value=True, key='show_bs')

            num_containers = st.sidebar.number_input(
                'Number of Customized Containers', 
                min_value=0, 
                max_value=4,
                value=1, 
                step=1,
                key='customized_containers')

    with st.expander("View RAG Pipeline Diagram"):
        st.image("image/RAGAv3.png", caption="Retrieval-Augmented Generation (RAG) Pipeline Diagram")
        

    ingestion_tab, retrieval_tab = st.tabs(["Ingestion", "Retrieval"])

    with ingestion_tab:
        if selected_column_option == 'Settings and Explain':
            col_settings, col_explain = st.columns([1, 2])
            # Retrieval_Pipeline.initialize_retrieval_instances_and_mappings()

            with col_settings:
                Ingestion_Settings.column_settings()

            with col_explain:
                Ingestion_Explain.column_explain()

            Ingestion_Display.column_display()
        
        elif selected_column_option == 'Display':
            col_display = st.columns([1])[0]

            with col_display:
                Ingestion_Display.column_display()




    with retrieval_tab:
        # if st.session_state.get('switch_to_display', False):
        #     st.session_state.selected_column_option = 'Display'
        #     st.session_state.switch_to_display = False

        if selected_column_option == 'All Columns':
            # Create columns for the three main sections
            col_settings, col_explain, col_display = st.columns([1.2, 2, 3])

            with col_settings:
                Retrieval_Settings.column_settings()

            with col_explain:
                Retrieval_Explain.column_explain()
            
            with col_display:
                Retrieval_Display.column_display()


        elif selected_column_option == 'Settings and Explain':
            col_settings, col_explain = st.columns([1, 2])

            with col_settings:
                Retrieval_Settings.column_settings()

            with col_explain:
                Retrieval_Explain.column_explain()

            Retrieval_Display.column_display()


        elif selected_column_option == 'Display':
            col_display = st.columns([1])[0]

            with col_display:
                Retrieval_Display.column_display()
                
        

    # st.write(st.session_state.setting_last_selection)
    # st.write(st.session_state.expanded_setting)
    st.write(st.session_state.vector_store)
    # st.write(st.session_state.customized_containers)
    # st.write(st.session_state.collection_name)
    # st.write(st.session_state.custom_collection_name)



if __name__ == "__main__":
    main()

