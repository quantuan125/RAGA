import streamlit as st
from textwrap import dedent
import re
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser 
from langchain.chat_models import ChatOpenAI


class Prompting:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    def baseline_prompt(self, combined_context, question):
        prompt_template = dedent("""
        
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: 
        {context}
        

        Question: {question}

        """)
        
        prompt_baseline = ChatPromptTemplate.from_template(prompt_template)
        #st.write(combined_context)

        # Now use the combined context to generate the final answer
        prompt_input = {
            "context": combined_context,
            "question": question
        }

        formatted_prompt = prompt_baseline.format(**prompt_input)
        formatted_prompt_baseline = re.sub(r'^Human:\s*', '', formatted_prompt)

        # Use the baseline prompt for generating the answer
        response_chain = prompt_baseline | self.llm | StrOutputParser()
        response = response_chain.invoke(prompt_input)
        # st.markdown("### Prompt:")
        # st.write(prompt_input)
        # st.markdown("### Answer:")
        # st.write(response)

        prompt_result = {
            "prompt": formatted_prompt_baseline,
            "response": response
            }

        return prompt_result
    
    def custom_prompt(self, combined_context, question):
        custom_template = st.session_state.get('custom_prompt_template', 'Default template if not set')

        prompt_custom = ChatPromptTemplate.from_template(custom_template)

        prompt_input = {
            "context": combined_context,
            "question": question
        }

        response_chain = prompt_custom | self.llm | StrOutputParser()
        response = response_chain.invoke(prompt_input)

        # st.write("Question:", question)
        # st.write("Answer:", response)
        return response
    
    def step_back_prompt(self, combined_context, question):
        step_back_template = """
        You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

        {context}

        Original Question: {question}
        Answer:
        """
        step_back_prompt = ChatPromptTemplate.from_template(step_back_template)

        prompt_input = {
            "context": combined_context,
            "question": question,  # Original question for generating the answer
        }
        
        # Generate the final answer using the combined context
        step_back_chain = step_back_prompt | self.llm | StrOutputParser()
        response = step_back_chain.invoke(prompt_input)

        # st.markdown("### Prompt:")
        # st.write(prompt_input)
        # st.markdown("### Answer:")
        # st.write(response)

        return response