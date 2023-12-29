import streamlit as st
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser 
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI


class QueryTransformer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    def multi_retrieval_query(self, question, **settings):
        query_count = settings.get('query_count', st.session_state.get('query_count', 3))
        mrq_prompt_template = settings.get(
            'mrq_prompt_template',
            st.session_state.get(
                'mrq_prompt_template', 
                """
                You are an AI language model assistant. Your task is to generate {query_count} different versions of the given user question to retrieve relevant documents from a vector database.
                
                Provide these alternative questions separated by newlines.
                Original question: {question}
                """
            )
        )

        formatted_prompt = mrq_prompt_template.format(query_count=query_count, question=question)
        mrq_prompt = ChatPromptTemplate.from_template(formatted_prompt)

        # Define the parser to split the LLM result into a list of queries
        def mrq_parse_queries(text):
            return text.strip().split("\n")

        # Chain the prompt with the LLM and the parser
        mrq_chain = mrq_prompt | self.llm | StrOutputParser() | mrq_parse_queries

        # Generate the multiple queries from the original question
        generated_queries = mrq_chain.invoke({"question": question})

        # st.markdown("### Queries:")
        # st.write(generated_queries)
        return generated_queries
    
    def query_extractor(self, question):

        class KeywordsSchema(BaseModel):
            keywords: str

            class Config:
                schema_extra = {
                    "description": "A schema to extract keywords from the given user query to improve vector search."
                }
        
        # Define the system message for the prompt
        system_message = """
        Extract and save relevant keywords mentioned in the following query together with their properties.
        
        If a property is not present and is not required in the function parameters, do not include it in the output.
        """

        # Create the extraction chain with the KeywordsSchema
        extraction_chain = create_extraction_chain_pydantic(KeywordsSchema, self.llm, system_message=system_message)
        #st.write(extraction_chain)

        # Invoke the chain with the original query
        result = extraction_chain.invoke({"input": question})
        #st.write(result)

        if result and isinstance(result, list) and len(result) > 0 and isinstance(result[0], KeywordsSchema):
            # Split keywords string from the first instance and rejoin them with a comma
            extracted_keywords = result[0].keywords.split(", ")
            extracted_query = ", ".join(extracted_keywords)

            # st.markdown("### Query Extractor:")
            # st.write(extracted_query)
            return extracted_query
        else:
            # If no valid result is found, return the original query
            return question
        
    def rewrite_retrieve_read(self, question):
        # Rewrite prompt template
        rewrite_template = """
        Provide a better search query for a vector search to answer the given question, end the query with ’**’.  
        Question: {question} 
        Answer:
        """

        # Create a prompt for rewriting the query
        rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)

        # Define the parser to remove '**'
        def parse_rewritten_query(text):
            return text.strip('**').strip()

        # Define the rewriter chain
        rewriter = rewrite_prompt | self.llm | StrOutputParser() | parse_rewritten_query

        rewritten_query = rewriter.invoke({"question": question})

        # st.markdown("### Rewritten Query")
        # st.write(rewritten_query)

        return rewritten_query
    
    def step_back_prompting(self, question):
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "What can the members of The Police do?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "What is Jan Sindel’s personal history?",
            },
            # Add more examples as needed
        ]

        # Step 2: Create the prompt template with examples
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=ChatPromptTemplate.from_messages(
                [("human", "{input}"), ("ai", "{output}")]
            )
        )

        step_back_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"),
                few_shot_prompt,
                ("user", "{question}"),
            ]
        )

        step_back_chain = step_back_prompt_template | self.llm | StrOutputParser()

        step_back_query = step_back_chain.invoke({"question": question})

        # st.markdown("### Step Back Query:")
        # st.write(step_back_query)

        step_back_and_original_query = [question, step_back_query]
        # st.write(step_back_and_original_query)

        return step_back_and_original_query