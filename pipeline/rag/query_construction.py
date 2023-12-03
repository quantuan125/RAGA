import streamlit as st
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.query_constructor.prompt import DEFAULT_SCHEMA
from langchain.prompts import ChatPromptTemplate
from langchain.chains.query_constructor.base import get_query_constructor_prompt, load_query_constructor_runnable
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.query_constructor.schema import AttributeInfo

class QueryConstructor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    def self_query_constructor(self, question):
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

        def format_toc_for_json(toc, indent_level=0):
            formatted_toc = ""
            indent = " " * (4 * indent_level)  # 4 spaces per indent level

            for entry in toc:
                # Iterate through each key-value pair in the entry
                for key, value in entry.items():
                    if key == "subsections":
                        if value:  # Only add subsections if they exist
                            formatted_toc += f"{indent}'{key}': [\n"
                            formatted_toc += format_toc_for_json(value, indent_level + 1)
                            formatted_toc += f"{indent}]\n"
                    else:
                        # Format the current section or subsection
                        formatted_toc += f"{indent}'{key}': '{value}',\n"

            return formatted_toc

        if st.session_state.toc_content:
            toc_string = format_toc_for_json(st.session_state.toc_content)
        else:
            toc_string = "No Table of Contents available."
        
        TOC_PROMPT = """\
        Use the Table of Contents (ToC) below to align the user's query with the document's structure. Analyze the user query and compare it against the ToC to identify the most relevant sections. Then, use this analysis to formulate precise filters targeting the headings (h1_main, h2_chapter, h3_subchapter) that are most likely to contain the information. 

        << Table of Contents >>
        \
        """

        custom_schema_prompt = DEFAULT_SCHEMA + "\n\n" + TOC_PROMPT + "\n\n" + toc_string
            
        query_constructor_prompt = get_query_constructor_prompt(
            st.session_state.document_content_description, 
            st.session_state.metadata_field_info,
            examples=examples,
            schema_prompt=custom_schema_prompt
        )

        formatted_constructor_prompt = query_constructor_prompt.format(query=question)
        formatted_constructor_prompt_html = formatted_constructor_prompt.replace("json<br>", "text<br>")
        st.markdown("### Query Constructor:")
        with st.expander("Query Constructor"):
            st.markdown(formatted_constructor_prompt_html, unsafe_allow_html=True)

        query_constructor_runnable = load_query_constructor_runnable(
            llm=self.llm,
            document_contents=st.session_state.document_content_description, 
            attribute_info=st.session_state.metadata_field_info,
            examples=examples,
            schema_prompt=custom_schema_prompt,
            fix_invalid=True
        )

        st.session_state.query_constructor = query_constructor_runnable

        structured_query = query_constructor_runnable.invoke({"query": question})
        st.markdown("### Structured Request Output:")
        st.json({"query": structured_query.query, "filter": str(structured_query.filter)})
        return structured_query

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
        

