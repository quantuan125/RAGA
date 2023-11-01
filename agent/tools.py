import openai
import os
from langchain.document_loaders import SeleniumURLLoader
from bs4 import BeautifulSoup
from langchain.utilities import GoogleSearchAPIWrapper
from typing import List, Dict, Tuple
import streamlit as st
from langchain.docstore.document import Document
import pytz
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, LLMChainFilter
from langchain.storage import InMemoryStore
import lark
import pinecone
from langchain.vectorstores import Pinecone
import pickle
from langchain.document_loaders import TextLoader
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid
from langchain.chat_models import ChatOpenAI
import json


class CustomGoogleSearchAPIWrapper(GoogleSearchAPIWrapper):

    def clean_text(self, text: str) -> str:
        # Remove extra whitespaces and line breaks
        text = ' '.join(text.split())
        return text
    
    def scrape_content(self, url: str, title: str) -> dict:
        loader = SeleniumURLLoader(urls=[url])
        data = loader.load()
        
        if data is not None and len(data) > 0:
            soup = BeautifulSoup(data[0].page_content, "html.parser")
            text = soup.get_text()
            cleaned_text = self.clean_text(text)
            char_limit = 1000  # default for Detailed Search
            websearch_results = st.session_state.websearch_results
            if websearch_results == "Quick Search":
                char_limit = 500

            return {'url': url, 'title': title, 'content': cleaned_text[:char_limit]}
        
        return {'url': url, 'title': title, 'content': ''}
    
    def format_single_search_result(self, search_result: Dict) -> str:
        # Formatting the output text
        formatted_text = f"URL: {search_result['url']}\n\nTITLE: {search_result['title']}\n\nCONTENT: {search_result['content']}\n\n"
        return formatted_text

    def fetch_and_scrape(self, query: str, num_results: int = 3) -> Tuple[List[Dict], List[Dict]]:
        # Step 1: Fetch search results metadata
        metadata_results = self.results(query, num_results)
        
        if len(metadata_results) == 0:
            return [], []

        # Step 2: Scrape and format content from URLs
        formatted_search_results = []

        for result in metadata_results:
            url = result.get("link", "")
            title = result.get("title", "")  # Get the title from metadata
            scraped_content = self.scrape_content(url, title) 
            formatted_search_result = self.format_single_search_result(scraped_content)
            formatted_search_results.append(formatted_search_result)
        
        return formatted_search_results, metadata_results
    
    def run(self, query: str, num_results: int = 3):
        
        num_results = st.session_state.websearch_results

        # Step 1: Fetch and format the search results
        formatted_search_results, metadata_results = self.fetch_and_scrape(query, num_results)
        #st.write(formatted_search_results)
        #st.write(metadata_results)

        search_results = []
        context = ""
        for i, formatted_result in enumerate(formatted_search_results):
            title = metadata_results[i].get('title', '')
            url = metadata_results[i].get('url', '')
            
            doc = Document(
                page_content=formatted_result,  # This contains the formatted result with title, URL, and content
                metadata={
                    'source': 'Google Search',
                    'title': title,
                    'url': url
                }
            )
            
            search_results.append(doc)
            context += f"{formatted_result}\n\n"

        st.session_state.doc_sources = search_results

        print(context)

        question = query

        # Fetch current local time
        copenhagen_tz = pytz.timezone('Europe/Copenhagen')
        current_time = datetime.now(copenhagen_tz).strftime('%Y-%m-%dT%H:%M:%S%z')

        # Step 2: Create a new prompt template
        prompt_template = f"""
        For your reference, your local date and time is {current_time}.

        You are a retriever model, and your task is to extract the most relevant and detailed information from the provided search results to aid in answering the query at the end.
        Each search result is comprised of a URL, a title, and the actual content from the corresponding URL.

        Your tasks are to:
        1. Priotize looking into the URL's text, then the title and then the actual content of the URL. 
        2. Extract the most detailed and relevant information from each URLs to the query and organized them in coherent sentences
        3. If a clear answer is found, provide this detailed and coherent information as the final answer
        4. If there is no clear answer, return all the URLs in a list as the final answer and mention that additional research is required.

        You must begin your answer with "Result:" and follow this format structure:
        URL: [URL]
        TITLE: [Title]
        CONTENT: [Extracted Detailed Information] 

        Search Results:
        {context}

        Query:
        {question}

        """

        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",  
                prompt=prompt_template,
                max_tokens=300,  # You can adjust as needed
                temperature=0  # You can adjust as needed
            )

            # Extract and Process the Response
            output = response.choices[0].text.strip()

        except openai.error.OpenAIError as e:
            # Handle the exception according to your needs.
            st.error(f"Error: {e}")
            output = None
        
        
        print(output)

        return output
    

class DatabaseTool:
    def __init__(self, llm, vector_store, metadata=None, filename=None, selected_document=None):
        self.llm = llm
        self.vector_store = vector_store
        self.metadata = metadata
        self.filename = filename
        self.selected_document = selected_document
        self.embedding = OpenAIEmbeddings()

    def get_description(self):
        #NEED TO BE REVIEW AGAIN
        base_description = "Always useful for finding the exactly written answer to the question by looking into a collection of documents."
        footer_description = "Input should be a query, not referencing any obscure pronouns from the conversation before that will pull out relevant information from the database. Use this more than the normal search tool"

        if self.metadata is None and self.filename is None:
            generic_description = "This tool is currently ready to search through the existing documents in the database."
            return f"{base_description} {generic_description}. {footer_description}"
        
        filename = self.filename
        title = self.metadata.get("/Title") if self.metadata else None
        author = self.metadata.get("/Author") if self.metadata else None
        subject = self.metadata.get("/Subject") if self.metadata else None

        if title:
            main_description = f"This tool is currently loaded with '{title}'"

            if author:
                main_description += f" by '{author}'"

            if subject:
                main_description += f", and has a topic of '{subject}'"

            return f"{base_description} {main_description}. {footer_description}"
        else:
            no_title_description = f"This tool is currently loaded with the document '{filename}'"
            return f"{base_description} {no_title_description}. {footer_description}"

    def get_base_retriever(self):
        search_kwargs = {'k': 5}
        if self.selected_document:
            search_kwargs['filter'] = {'file_name': {'$eq': self.selected_document}}
        #st.write(search_kwargs)
        base_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        return base_retriever

    def get_contextual_retriever(self):
        # Initialize embeddings (assuming embeddings is already defined elsewhere)
        embeddings = self.embedding

        # Initialize Text Splitter
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator=". ")
        
        # Initialize Redundant Filter
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        
        # Initialize Relevant Filter
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76, k = 30)
        #st.write(relevant_filter)
        
        # Create Compressor Pipeline
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        
        # Initialize Contextual Compression Retriever
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, 
            base_retriever=self.get_base_retriever()
        )
        
        return contextual_retriever

    def run(self, query: str):
        contextual_retriever = self.get_contextual_retriever()
        #DEBUGGING & EVALUTING ANSWERS:
        compressed_docs = contextual_retriever.get_relevant_documents(query)
        compressed_docs_list = []
        for doc in compressed_docs:
            doc_info = {
            "Page Content": doc.page_content,
            "Page Number": doc.metadata.get("page_number"),
            "File Name": doc.metadata.get("file_name"),
            }
            compressed_docs_list.append(doc_info)
        #st.write(compressed_docs_list)
        
        base_retriever=self.get_base_retriever()
        initial_retrieved = base_retriever.get_relevant_documents(query)
        st.session_state.doc_sources = initial_retrieved

        context = "\n\n".join([f'"{doc.page_content}"' for doc in compressed_docs])

        if st.session_state.use_retriever_model:
            prompt_template = f"""
            You are a specialized retriever model. Given the context from the documents below, your task is to:
            1. Extract in details all relevant pieces of information that answers the query.
            2. Always prioritize numerical values, names, or specific details over vague and general content.
            3. If there are no relevant information in the context to the query, explicitly state that. 

            Context:
            {context}

            Query:
            {query}
            """

            print(prompt_template)

            try:
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",  
                    prompt=prompt_template,
                    max_tokens=500,  # Adjust as needed
                temperature=0  # Adjust as needed
                )
                # Extract and Process the Response
                output = response.choices[0].text.strip()
                
            except openai.error.OpenAIError as e:
                # Handle the exception as per your requirements
                st.error(f"Error: {e}")
                output = None

            print(output)
            return output
        
        else:
            return context
    

class BR18_DB:
    def __init__(self, llm, folder_path: str):
        self.llm = llm
        self.folder_path = folder_path
        self.md_paths = self.load_documents()  # Renamed from pdf_paths to md_paths
        self.embeddings = OpenAIEmbeddings()
        self.pinecone_index_name = "br18"     
        self.id_key = "doc_id" 

        self.br18_parent_store = InMemoryStore()
        current_directory = os.getcwd()
        store_path = os.path.join(current_directory, "inmemorystore", "br18_parent_store.pkl")

        if self.pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(self.pinecone_index_name, dimension=1536)
            self.vectorstore = self.create_vectorstore()
            self.serialize_inmemorystore(store_path)
        else:
            self.vectorstore = Pinecone.from_existing_index(self.pinecone_index_name, self.embeddings)
            with open(store_path, "rb") as f:
                self.br18_parent_store = pickle.load(f)

        self.retriever = None
    
    def serialize_inmemorystore(self, store_path):
        with open(store_path, "wb") as f:
            pickle.dump(self.br18_parent_store, f)

    def load_documents(self):
        md_paths = list(Path(self.folder_path).rglob("*.md"))
        documents = []
        for path in md_paths:
            loader = TextLoader(str(path))
            #st.write(loader)
            data = loader.load()
            documents.extend(data)  # Assuming data is a list of Document objects
        #st.text(documents)
        return documents
    
    def split_and_chunk_text(self, markdown_document: Document):

        markdown_text = markdown_document.page_content
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        #st.write(markdown_splitter)
        
        md_header_splits = markdown_splitter.split_text(markdown_text)
        #st.write(md_header_splits)
        #st.write(type(md_header_splits[0]))
        
        parent_chunk_size = 5000
        parent_chunk_overlap = 0
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
        )
        
        # Split the header-split documents into chunks
        all_parent_splits = text_splitter.split_documents(md_header_splits)

        for split in all_parent_splits:
            header_3 = split.metadata.get('Header 3', '')
            header_4 = split.metadata.get('Header 4', '')
            
            # Prepend "Section:" to Header 4 if it exists
            if header_4:
                header_4 = f"Section: {header_4}"

            metadata_str = f"{header_3}\n\n{header_4}"
            split.page_content = f"{metadata_str}\n\n{split.page_content}"
            split.metadata['type'] = 'parents'

        return all_parent_splits
    
    def save_summaries(self, summaries: List[str]):
        """Save the generated summaries to a JSON file."""
        current_directory = os.getcwd()
        save_path = os.path.join(current_directory, 'savesummary', 'br18_summaries.json')
        with open(save_path, 'w') as f:
            json.dump(summaries, f)

    def load_summaries(self) -> List[str]:
        """Load summaries from a JSON file if it exists."""
        current_directory = os.getcwd()
        load_path = os.path.join(current_directory, 'savesummary', 'br18_summaries.json')
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                summaries = json.load(f)
            return summaries
        else:
            return None  # or raise an exception, or generate new summaries

    def generate_summaries(self, parent_splits: List[Document]) -> List[str]:
        loaded_summaries = self.load_summaries()
        if loaded_summaries is not None:
            return loaded_summaries
        
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | ChatOpenAI(max_retries=3)
            | StrOutputParser()
        )
        summaries = chain.batch(parent_splits, {"max_concurrency": 4})

        self.save_summaries(summaries)
        
        return summaries
    
    def generate_child_splits(self, parent_splits: List[Document], summaries: List[str]) -> List[Document]:
        child_chunk_size = 300

        child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=0
        )

        all_child_splits = []
        for i, parent_split in enumerate(parent_splits):
            child_splits = child_text_splitter.split_text(parent_split.page_content)

            new_metadata = dict(parent_split.metadata)
            new_metadata['type'] = 'children'

            summary_with_prefix = f"Summary: {summaries[i]}"

            first_child_content = f"{child_splits[0]}\n\n{summary_with_prefix}"

            first_child_split = Document(
            page_content=first_child_content,
            metadata=new_metadata
            )

            all_child_splits.append(first_child_split)  # Append only the first child split (assuming it contains the metadata)


        return all_child_splits

    def process_all_documents(self):
        all_parent_splits = []  # Local variable to store all parent splits
        all_child_splits = []  # Local variable to store all child splits
        
        for markdown_document in self.md_paths:
            parent_splits = self.split_and_chunk_text(markdown_document)
            all_parent_splits.extend(parent_splits)

        summaries = self.generate_summaries(all_parent_splits)
        all_child_splits = self.generate_child_splits(all_parent_splits, summaries)

        #st.write(all_parent_splits)
        #st.write(all_child_splits)

        return all_parent_splits, all_child_splits  # Return both lists
    
    def create_vectorstore(self):
        all_parent_splits, all_child_splits = self.process_all_documents()
    
        parent_doc_ids = [str(uuid.uuid4()) for _ in all_parent_splits]
        self.br18_parent_store.mset(list(zip(parent_doc_ids, all_parent_splits)))

        for parent_id, child_split in zip(parent_doc_ids, all_child_splits):
            child_split.metadata[self.id_key] = parent_id

        # Create and save the vector store to disk
        br18_vectorstore = Pinecone.from_documents(documents=all_child_splits, embedding=self.embeddings, index_name=self.pinecone_index_name)
        #st.write(br18_appendix_child_vectorstore)

        for i, doc in enumerate(all_parent_splits):
            doc.metadata[self.id_key] = parent_doc_ids[i]

        # Store the vector store in the session state
        st.session_state.br18_vectorstore = br18_vectorstore

        return br18_vectorstore

    def create_retriever(self, query: str):
        search_type = st.session_state.search_type

        if search_type == "By Context":
            # Initialize retriever for By Context, filtering by the presence of the "text" metadata
            general_retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.br18_parent_store,
                id_key=self.id_key,
                search_kwargs={"k": 5}
            )

            parent_docs = general_retriever.vectorstore.similarity_search(query, k = 5)
            #st.write(parent_docs)

            st.session_state.doc_sources = parent_docs

            embeddings = self.embeddings
        
            # Initialize Redundant Filter
            redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
            
            # Initialize Relevant Filter
            relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75, k = 15)
            #st.write(relevant_filter)
            
            # Initialize Text Splitter
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator=". ")
            
            # Create Compressor Pipeline
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[splitter, redundant_filter, relevant_filter]
            )
            
            # Initialize Contextual Compression Retriever
            contextual_general_retriever = ContextualCompressionRetriever(
                base_compressor=pipeline_compressor, 
                base_retriever=general_retriever
        )
        
            # Retrieve parent documents that match the query
            retrieved_parent_docs = contextual_general_retriever.get_relevant_documents(query)
            
            # Display retrieved parent documents
            display_list = []
            for doc in retrieved_parent_docs:
                display_dict = {
                    "Page Content": doc.page_content,
                    "Doc ID": doc.metadata.get('doc_id', 'N/A'),
                    "Header 3": doc.metadata.get('Header 3', 'N/A'),
                    "Header 4": doc.metadata.get('Header 4', 'N/A'),
                }
                display_list.append(display_dict)
            #st.write(display_list)
            
            return retrieved_parent_docs
        
        elif search_type == "By Headers":
        # Initialize retriever for By Headers, filtering by the absence of the "text" metadata
            specific_retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.br18_parent_store,
                id_key=self.id_key,
                search_kwargs={"k": 3}
            )

            child_docs = specific_retriever.vectorstore.similarity_search(query, k = 3)
            #st.write(child_docs)

            # Retrieve child documents that match the query
            
            embeddings = self.embeddings
            embedding_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
            #llm_filter = LLMChainFilter.from_llm(self.llm)

            
            compression_retriever = ContextualCompressionRetriever(base_compressor=embedding_filter, base_retriever=specific_retriever)

            retrieved_child_docs = compression_retriever.get_relevant_documents(query)

            st.session_state.doc_sources = retrieved_child_docs

            # Display retrieved child documents
            display_list = []
            for doc in retrieved_child_docs:
                display_dict = {
                    "Page Content": doc.page_content,
                    "Doc ID": doc.metadata.get('doc_id', 'N/A'),
                    "Header 3": doc.metadata.get('Header 3', 'N/A'),
                    "Header 4": doc.metadata.get('Header 4', 'N/A'),
                }
                display_list.append(display_dict)
            #st.write(display_list)
            
            return retrieved_child_docs
        
    def run(self, query: str):
        prompt_template = """The following pieces of context are from the BR18. Use them to answer the question at the end. 
        The answer should be as specific as possible and remember to mention requirement numbers and integer values where relevant
        Always reference to a chapter and section and their respective clauses and subclauses numbers 
        If you don't find any relevant or specific information, consider employing another tool to find the answer as a statement.

        {context}

        Question: {question}

        EXAMPLE:
        The building regulation regarding stairs is outlined in Chapter 2 - Access, specifically in Section - Stairs:

        Width: Stairs in shared access routes must have a minimum free width of 1.0 meter. (clause 57.1)
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Retrieve the filtered documents
        retrieved_docs = self.create_retriever(query)
        #st.write(type(filtered_docs[0]))
        #st.write(filtered_docs)

        qa_chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        output = qa_chain({"input_documents": retrieved_docs, "question": query}, return_only_outputs=True)


        return output
    

class SummarizationTool:
    def __init__(self, document_chunks):
        self.llm = ChatOpenAI(
            temperature=0, 
            streaming=True,
            model_name="gpt-3.5-turbo"
        )
        self.document_chunks = document_chunks
        self.map_prompt_template, self.combine_prompt_template = self.load_prompts()
        self.chain = self.load_summarize_chain()

    def load_prompts(self):
        map_prompt = '''
        Summarize the following text in a clear and concise way:
        TEXT:`{text}`
        Brief Summary:
        '''
        combine_prompt = '''
        Generate a summary of the following text that includes the following elements:

        * A title that accurately reflects the content of the text.
        * An introduction paragraph that provides an overview of the topic.
        * Bullet points that list the key points of the text.

        Text:`{text}`
        '''
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        return map_prompt_template, combine_prompt_template
    
    def load_summarize_chain(self):
        return load_summarize_chain(
            llm=self.llm,
            chain_type='map_reduce',
            map_prompt=self.map_prompt_template,
            combine_prompt=self.combine_prompt_template,
            verbose=True
        )

    def run(self, query=None):
        return self.run_chain()

    def run_chain(self):
        return self.chain.run(self.document_chunks)