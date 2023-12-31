import streamlit as st
import json
import os
import re
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.query_constructor.prompt import DEFAULT_SCHEMA
from langchain.prompts import ChatPromptTemplate
from langchain.chains.query_constructor.base import get_query_constructor_prompt, load_query_constructor_runnable
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_core.runnables import RunnablePassthrough
class QueryConstructor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

    def format_toc_for_json(toc, indent_level=0):
        formatted_toc = ""
        indent = " " * (4 * indent_level)  # 4 spaces per indent level

        for entry in toc:
            for key, value in entry.items():
                if key == "subsections":
                    if value:
                        formatted_toc += f"{indent}'{key}': [\n"
                        formatted_toc += QueryConstructor.format_toc_for_json(value, indent_level + 1)
                        formatted_toc += f"{indent}]\n"
                else:
                    formatted_toc += f"{indent}'{key}': '{value}',\n"

        return formatted_toc

    def self_query_constructor(self, question, **settings):

        metadata_field_info = settings.get('metadata_field_info', st.session_state.get('metadata_field_info', None))
        toc_content = settings.get('toc_content', st.session_state.get('toc_content', None))
        document_content_description = settings.get('document_content_description', st.session_state.get('document_content_description', None))

        examples = [
            (
                "Can you tell me about the regulations for stairs?",
                {
                "query": "stairs",
                "filter": "contain(\"h3_subchapter\", \"Stairs\")"
                }
            ),
            (
                "What are the general requirements regarding fire?",
                {
                    "query": "fire",
                    "filter": "contain(\"h2_chapter\", \"Chapter 5 Fire\")"
                }
            ),
            (
                "What is needed for a building permit?",
                {
                "query": "building permit",
                "filter": "contain(\"h3_subchapter\", \"Building permit\")"
                }
            ),
            (
                "What are regulation for fences and handrails?",
                {
                "query": "fences, handrails",
                "filter": "or(contain(\"h3_subchapter\", \"Fencing\"), contain(\"h3_subchapter\", \"Hand rails\"))"
                }
            ),
        ]

        if toc_content:
            toc_string = QueryConstructor.format_toc_for_json(toc_content)
        else:
            toc_string = "No Table of Contents available."
        
        TOC_PROMPT = """\
        Use the Table of Contents (ToC) below to align the user's query with the document's structure. Analyze the user query and compare it against the ToC to identify the most relevant sections. Then, use this analysis to formulate precise filters targeting the headings (h1_main, h2_chapter, h3_subchapter) that are most likely to contain the information. 

        << Table of Contents >>
        \
        """

        custom_schema_prompt = DEFAULT_SCHEMA + "\n\n" + TOC_PROMPT + "\n\n" + toc_string
            
        query_constructor_prompt = get_query_constructor_prompt(
            document_content_description, 
            metadata_field_info,
            examples=examples,
            schema_prompt=custom_schema_prompt
        )

        formatted_constructor_prompt = query_constructor_prompt.format(query=question)
        formatted_constructor_prompt_html = formatted_constructor_prompt.replace("json<br>", "text<br>")
        # st.markdown("### Query Constructor:")
        # with st.expander("Query Constructor"):
        #     st.markdown(formatted_constructor_prompt_html, unsafe_allow_html=True)

        query_constructor_runnable = load_query_constructor_runnable(
            llm=self.llm,
            document_contents=document_content_description, 
            attribute_info=metadata_field_info,
            examples=examples,
            schema_prompt=custom_schema_prompt,
            fix_invalid=True
        )

        st.session_state.query_constructor = query_constructor_runnable

        structured_query = query_constructor_runnable.invoke({"query": question})
        # st.markdown("### Structured Request Output:")
        # st.json({"query": structured_query.query, "filter": str(structured_query.filter)})
        
        qc_result = {
            'structured_query': structured_query,
            'constructor_prompt': formatted_constructor_prompt_html
        }
        return qc_result

    def generate_metadata_descriptions(self, field_name, header_info_list):
        header_info_tuple = [(h[0], h[1]) for h in header_info_list]
        
        header_info = '[' + ', '.join([f"('{h[0]}', '{h[1]}')" for h in header_info_tuple]) + ']'

        generate_metadata_description = """
        "You are an expert in Markdown file structure and syntax. Given a list of tuples indicating all possible metadata fields for this document, provide a brief description (15 words or less) of the specified field including its position in the hierarchy in relation to each of the markdown header syntax's field in <all fields>

        Your answer should always start with: The "<specified field>" represents

        all fields: {header_info}
        specified field: {field_name}
        """

        metadata_description_prompt_template = ChatPromptTemplate.from_template(generate_metadata_description)

        prompt_input = {
        "header_info": header_info,
        "field_name": field_name
        }
        chain = metadata_description_prompt_template | self.llm | StrOutputParser()

        response = chain.invoke(prompt_input)

        return response
    
    def build_metadata_field_info(self, schema, header_info):
        attr_info_list = []

        base_filter_instructions = (
            "ALWAYS filter with one or more CONTAINS comparators, "
            "and use the OR operator to check ALL other fields. "
            "If the value of this field contains a word or phrase that is very similar to a word or phrase in the query, "
            "filter for the exact string from the value rather than the query. "
        )

        # Instructions for prioritizing Header 3 in filters
        header1_filter_instructions = (
            "Generally avoid using this field for filtering unless the queries specifically asked about 'appendix' section."\
            "the Header 1-level filter should ALWAYS be combined with subsection filters using an AND operator. \n"
        )

        # Instructions for using Header 2, especially for chapters that broadly cover a topic
        header2_filter_instructions = (
            "Use this field for broader queries related to chapters. "
            "It is especially useful when the query includes general terms."
        )

        # Instructions for using Header 3, which should be prioritized for specific queries
        header3_filter_instructions = (
            "Prioritize this field for specific and detailed queries"
            "Combine with Header 2 for detailed filtering and complete context."
        )

        for field in schema["text"]:
            # Generate a description for each field
            desc = self.generate_metadata_descriptions(field["name"], header_info) + base_filter_instructions

            # if "Header 3" in field["name"]:
            #     desc += header3_filter_instructions
            # elif "Header 2" in field["name"]:
            #     desc += header2_filter_instructions
            # elif "Header 1" in field["name"]:
            #     desc += header1_filter_instructions

            # Create the AttributeInfo object
            attr_info = AttributeInfo(
                name=field["name"],
                description=desc,
                type="string"
            )
            attr_info_list.append(attr_info)

        return attr_info_list
    
    def sql_query_constructor(self, question):
        template = """You are a SQLite expert. Given an input question, you must create a syntactically correct SQLite query to run.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        Pay attention to use date('now') function to get the current date, if the question involves "today".

        #Use the following format:

        Question: <Question here>
        SQLQuery: <SQL Query to run>

        #Only use the following tables:

        {schema}.

        #INSTRUCTION:
        Your answer must begin with "SQLQuery:" and you are not allowed to answer in natural language:

        #EXAMPLE:
        QUESTION: What is the order with the longest delay?
        SQLQuery: SELECT "OrderID", MAX("TimeCompleted" - "TimeSent") as "LongestDelay" FROM orders;

        QUESTION: What is the heaviest order and how heavy is it? 
        SQLQuery: SELECT "OrderID", "WeighingSlip" FROM orders WHERE "WeighingSlip" IS NOT NULL ORDER BY "WeighingSlip" DESC LIMIT 1;

        #QUESTION: 
        {question}
        """

        sql_schema = st.session_state.database.get_table_info()

        sql_constructor_prompt = ChatPromptTemplate.from_template(template)

        formatted_sql_constructor_prompt = sql_constructor_prompt.format_messages(question=question, top_k=5, schema=sql_schema)

        prompt_input = {
            "schema": sql_schema,
            "question": question,
            "top_k": 5
        }

        def remove_sql_query_prefix(text):
            # Remove 'SQLQuery:' prefix and strip leading/trailing whitespace
            return text.replace("SQLQuery:", "").strip()

        sql_query_chain = (
            sql_constructor_prompt
            | self.llm.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
            | remove_sql_query_prefix
        )

        st.session_state.query_constructor = sql_query_chain

        sql_query_response = sql_query_chain.invoke(prompt_input)

        st.session_state.sql_query = sql_query_response

        sql_query_result = {
            'structured_query': sql_query_response,
            'constructor_prompt': formatted_sql_constructor_prompt
        }

        return sql_query_result

    def sql_semantic_query_constructor(self, question):
        template = """You are a Postgres expert. Given an input question, you must create a syntactically correct Postgres query to run.

        Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per Postgres. You can order the results to return the most informative data in the database.
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

        sql_schema = st.session_state.database.get_table_info()

        sql_constructor_prompt = ChatPromptTemplate.from_template(template)

        formatted_sql_constructor_prompt = sql_constructor_prompt.format_messages(question=question, top_k=5, schema=sql_schema)

        prompt_input = {
            "schema": sql_schema,
            "question": question,
            "top_k": 5
        }

        def remove_sql_query_prefix(text):
            # Remove 'SQLQuery:' prefix and strip leading/trailing whitespace
            return text.replace("SQLQuery:", "").strip()

        sql_query_chain = (
            sql_constructor_prompt
            | self.llm.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
            | remove_sql_query_prefix
        )

        st.session_state.query_constructor = sql_query_chain

        sql_query = sql_query_chain.invoke(prompt_input)

        def replace_brackets(match):
            words_inside_brackets = match.group(1).split(", ")
            embedded_words = [
                str(OpenAIEmbeddings().embed_query(word)) for word in words_inside_brackets
            ]
            return "', '".join(embedded_words)

        def get_query(query):
            sql_query = re.sub(r"\[([\w\s,]+)\]", replace_brackets, query)
            return sql_query
        
        embedded_sql_query = get_query(sql_query)
        
        st.session_state.sql_query = embedded_sql_query

        sql_query_result = {
            'structured_query': sql_query,
            'constructor_prompt': formatted_sql_constructor_prompt
        }

        return sql_query_result
