import streamlit as st
from dotenv import load_dotenv
from UI.css import apply_css
from utility.sessionstate import Init
import langchain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.vectorstores.redis import Redis
import json
from langchain.embeddings import OpenAIEmbeddings
from streamlit_elements import elements, mui, html
from langchain.chat_models import ChatOpenAI
from langchain.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough
import re
from langchain_core.runnables import RunnableLambda



langchain.debug=True


def main():
    load_dotenv()
    st.set_page_config(page_title="Playground", page_icon="🕹️", layout="wide")
    apply_css()
    st.title("PLAYGROUND 🕹️")
    
    with st.empty():
        Init.initialize_session_state()
        # Init.initialize_agent_state()
        # Init.initialize_clientdb_state()

    # db_path = "sqlite:///storage/SQL/Chinook.db"
    db_path = "postgresql+psycopg2://postgres:Theredpill123!@localhost:5432/chinook_pg_main"
    db = SQLDatabase.from_uri(db_path)
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    # db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    embeddings_model = OpenAIEmbeddings()

    def get_schema(_):
        return db.get_table_info()

    def run_query(query):
        return db.run(query)
    
    def replace_brackets(match):
        words_inside_brackets = match.group(1).split(", ")
        embedded_words = [
            str(embeddings_model.embed_query(word)) for word in words_inside_brackets
        ]
        return "', '".join(embedded_words)


    def get_query(query):
        sql_query = re.sub(r"\[([\w\s,]+)\]", replace_brackets, query)
        return sql_query
    
    def remove_sql_query_prefix(text):
        # Remove 'SQLQuery:' prefix and strip leading/trailing whitespace
        return text.replace("SQLQuery:", "").strip()
        
    template = """You are a Postgres expert. Given an input question, first create a syntactically correct Postgres query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per Postgres. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use date('now') function to get the current date, if the question involves "today".

    You can use an extra extension which allows you to run semantic similarity using <-> operator on tables containing columns named "embeddings".
    <-> operator can ONLY be used on embeddings columns.
    The embeddings value for a given row typically represents the semantic meaning of that row.
    The vector represents an embedding representation of the question, given below. 
    Do NOT fill in the vector values directly, but rather specify a `[search_word]` placeholder, which should contain the word that would be embedded for filtering.
    For example, if the user asks for songs about 'the feeling of loneliness' the query could be:
    'SELECT "[whatever_table_name]"."SongName" FROM "[whatever_table_name]" ORDER BY "embeddings" <-> '[loneliness]' LIMIT 5'

    Your answer must be a in SQL format and you are not allowed to answer in natural language:

    Use the following format:

    Question: <Question here>
    SQLQuery: <SQL Query to run>
    SQLResult: <Result of the SQLQuery>
    Answer: <Final answer here>

    Only use the following tables:

    {schema}

    EXAMPLE 1:
    Question: Which are the 10 rock songs with titles about deep feeling of loneliness?
    SQLQuery: SELECT "Track"."Name" FROM "Track" JOIN "Genre" ON "Track"."GenreId" = "Genre"."GenreId" WHERE "Genre"."Name" = \'Rock\' ORDER BY "Track"."embeddings" <-> \'[dispair]\' LIMIT 10

    EXAMPLE 2:
    Question: I need the 6 albums with shortest title, as long as they contain songs which are in the 20 saddest song list.
    SQLQuery: WITH "SadSongs" AS (SELECT "TrackId" FROM "Track" ORDER BY "embeddings" <-> '[sad]' LIMIT 20),"SadAlbums" AS (SELECT DISTINCT "AlbumId" FROM "Track" WHERE "TrackId" IN (SELECT "TrackId" FROM "SadSongs"))SELECT "Album"."Title" FROM "Album" WHERE "AlbumId" IN (SELECT "AlbumId" FROM "SadAlbums") ORDER BY "title_len" ASC LIMIT 6


    QUESTION: {question}
    SQLQuery:
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", template)]
    )

    sql_query_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | remove_sql_query_prefix
    )

    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", template), ("human", "{question}")]
    )

    full_chain = (
    RunnablePassthrough.assign(query=sql_query_chain)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=RunnableLambda(lambda x: db.run(get_query(x["query"]))),
    )
    | prompt
    | llm
    | StrOutputParser()
    )
    
    st.subheader("Text-SQL + semantic")

    # UI for query input
    user_query = st.text_input("Enter your query:", placeholder="E.g., How many employees are there?")
    
    if st.button("Run Query"):
        # Process query and display result
        try:
            sql_query = full_chain.invoke({"question": user_query})
            # sql_query = full_chain.invoke({"question": user_query})
            st.write("Generated SQL Query:", sql_query)
            st.session_state.result = sql_query
        except Exception as e:
            st.error(f"Error executing query: {e}")

    st.subheader("Text-SQL")
    
    sql_query = st.text_input("Enter your SQL query:", placeholder="E.g., SELECT * FROM \"Track\" LIMIT 5;")
    
    if st.button("Run SQL Query"):
        # Execute SQL query and display result
        try:
            result = db.run(sql_query)
            st.session_state.result = result
            st.write(result)
            st.success("SQL Query executed successfully.")
        except Exception as e:
            st.error(f"Error executing SQL query: {e}")
    
    with st.expander("Answer"):
        st.write(st.session_state.result)

    st.subheader("Test - Text-SQL + semantic")

    input_phrase = st.text_input("Enter a phrase for semantic search:", placeholder="E.g., hope about the future")

    if st.button("Run Semantic Query"):
        # Generate the embedding for the input phrase
        if input_phrase:
            try:
                embeded_title = embeddings_model.embed_query(input_phrase)

                # Construct the query using the embedding
                semantic_query = (
                    'SELECT "Track"."Name" FROM "Track" WHERE "Track"."embeddings" IS NOT NULL '
                    'ORDER BY "embeddings" <-> ' + f"'{embeded_title}' LIMIT 5"
                )

                # Execute the query and display result
                result = db.run(semantic_query)
                st.write("Semantic Query Result:", result)
                st.write(semantic_query)
                st.success("Semantic Query executed successfully.")
            except Exception as e:
                st.error(f"Error executing semantic query: {e}")
        else:
            st.error("Please enter a phrase for the semantic search.")
    
    with st.expander("Generate and Insert Embeddings"):
        generate_track_embeddings = st.checkbox("Generate embeddings for 'Track' table")
        generate_album_embeddings = st.checkbox("Generate embeddings for 'Album' table")

        if st.button("Generate and Insert Embeddings"):
            # Process 'Track' table if checked
            if generate_track_embeddings:
                try:
                    tracks = db.run('SELECT "Name" FROM "Track"')
                    song_titles = [s[0] for s in eval(tracks)]
                    title_embeddings = embeddings_model.embed_documents(song_titles)
                    
                    for i, (title, embedding) in enumerate(zip(song_titles, title_embeddings)):
                        title = title.replace("'", "''")  # Escape single quotes
                        sql_command = f'UPDATE "Track" SET "embeddings" = ARRAY{embedding} WHERE "Name" =\'{title}\''
                        db.run(sql_command)

                    st.success("Embeddings for 'Track' table generated and inserted successfully.")
                except Exception as e:
                    st.error(f"Error processing 'Track' table: {e}")

            # Process 'Album' table if checked
            if generate_album_embeddings:
                try:
                    db.run('ALTER TABLE "Album" ADD COLUMN IF NOT EXISTS "embeddings" vector;')
                    albums = db.run('SELECT "Title" FROM "Album"')
                    album_titles = [a[0] for a in eval(albums)]
                    album_embeddings = embeddings_model.embed_documents(album_titles)
                    
                    for i, (title, embedding) in enumerate(zip(album_titles, album_embeddings)):
                        title = title.replace("'", "''")  # Escape single quotes
                        sql_command = f'UPDATE "Album" SET "embeddings" = ARRAY{embedding} WHERE "Title" =\'{title}\''
                        db.run(sql_command)

                    st.success("Embeddings for 'Album' table generated and inserted successfully.")
                except Exception as e:
                    st.error(f"Error processing 'Album' table: {e}")

    # try:
    #     db_path = "sqlite:///C:/Users/quant/OneDrive/Documents/Purpose/AI/RAGA/main/storage/SQL/Chinook.db"
    #     db = SQLDatabase.from_uri(db_path)
    #     results = db.run("SELECT * FROM Artist LIMIT 10;")
    #     st.success("Connected to Chinook database successfully!")
    #     st.write("Sample data from the Artist table:")
    #     st.write(results)
    # except Exception as e:
    #     st.error(f"Failed to connect to the Chinook database: {e}")



if __name__ == "__main__":
    main()