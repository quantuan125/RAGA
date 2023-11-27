import streamlit as st
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.query_constructor.prompt import DEFAULT_SCHEMA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.redis import RedisTranslator
from langchain.vectorstores.redis.schema import RedisModel
from langchain.chains.query_constructor.base import StructuredQueryOutputParser, get_query_constructor_prompt


class QueryConstructor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    def self_query_constructor(self):
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
                "filter": "or(contain(\"h3_subchapter\", \"Stairs\"), contain(\"h3_subchapter\", \"Hand rails\"))"
                }
            ),
        ]

        def format_toc_for_json(toc):
            formatted_toc = json.dumps(toc, indent=4)
            # Escape curly braces for JSON format
            formatted_toc = formatted_toc.replace("{", "{{{{").replace("}", "}}}}")
            # Add escape character at the beginning and end of the ToC string
            return formatted_toc 

        toc_file_path = os.path.join('json', 'toc.json')
        toc_string = """"""
        if os.path.exists(toc_file_path):
            with open(toc_file_path, 'r') as file:
                toc = json.load(file)
                toc_string = format_toc_for_json(toc)
        
        TOC_PROMPT = """\
        Use the Table of Contents (ToC) below to align the user's query with the document's structure. Analyze the user query and compare it against the ToC to identify the most relevant sections. Then, use this analysis to formulate precise filters targeting the headings (h1_main, h2_chapter, h3_subchapter) that are most likely to contain the information. 

        << Table of Contents >>
        \
        """

        custom_schema_prompt = DEFAULT_SCHEMA + "\n\n" + TOC_PROMPT + "\n\n" + toc_string
            
        constructor_prompt = get_query_constructor_prompt(
            st.session_state.document_content_description, 
            st.session_state.metadata_field_info,
            examples=examples,
            schema_prompt=custom_schema_prompt
            )
        
        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = constructor_prompt | self.llm | output_parser

        index_schema = st.session_state.index_schema
        redis_schema = RedisModel(**index_schema)

        sqr_retriever = SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=st.session_state.vector_store,
            structured_query_translator=RedisTranslator(schema=redis_schema),
        )

        print(constructor_prompt.input_variables)
        formatted_constructor_prompt = constructor_prompt.format(query="")
        formatted_constructor_prompt_html = formatted_constructor_prompt.replace("json<br>", "text<br>")

        st.markdown("### Query Constructor:")
        with st.expander("Query Constructor"):
            st.markdown(formatted_constructor_prompt_html, unsafe_allow_html=True)
        
        return sqr_retriever
