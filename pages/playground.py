from UI.sidebar import Sidebar
from rag.query_transformation import QueryTransformer
from rag.query_construction import QueryConstructor
from rag.vector_search import VectorSearch
from rag.post_processing import PostProcessor
import streamlit as st
from dotenv import load_dotenv
from UI.css import apply_css
from utility.sessionstate import Init
from UI.main import Main
from agent.tools import DatabaseTool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser 
from langchain.load import dumps, loads
from langchain.indexes import VectorstoreIndexCreator
import langchain
from langchain.schema import format_document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    LongContextReorder
)
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import CharacterTextSplitter
import tiktoken
from langchain.schema import Document
from typing import List
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain.pydantic_v1 import BaseModel
from langchain.document_loaders import TextLoader
import pypdf
from pathlib import Path
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import tempfile
import os
from unstructured.partition.md import partition_md
from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes, clean_bullets, clean_trailing_punctuation, clean_ordered_bullets
import uuid
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.vectorstores.redis import Redis
import json
from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    construct_examples,
    AttributeInfo)

from langchain.chains.query_constructor.prompt import (
    DEFAULT_SCHEMA
)
from langchain.chains.query_constructor.ir import (
    Comparator,
    Operator,
)

langchain.debug=True
def display_documents(docs_list):
    formatted_docs = []
    for doc in docs_list:
        if isinstance(doc, Document):
            doc_info = {
                "Page Content": doc.page_content,
                # Add other relevant fields here if needed
            }
            formatted_docs.append(doc_info)
        else:
            formatted_docs.append("Not a Document object")

    st.write(formatted_docs)

def truncate_to_token_limit(text, max_tokens, model_name):
    # Retrieve the correct encoding for the model
    encoding = tiktoken.encoding_for_model(model_name)

    # Encode the text and get the number of tokens
    tokens = encoding.encode(text)
    num_tokens = len(tokens)

    # If the number of tokens is within the limit, return the text as is
    if num_tokens <= max_tokens:
        return text
    
    # If the number of tokens exceeds the limit, truncate
    truncated_text = encoding.decode(tokens[:max_tokens])
    return truncated_text

def create_combined_context(retrieval_results, max_tokens=15500, model_name='gpt-3.5-turbo'):
    # Combine the page_content of each Document into a single string
    #st.write(retrieval_results)
    combined_context = "\n".join([doc.page_content for doc in retrieval_results])

    # Ensure the combined context is within the token limit
    combined_context = truncate_to_token_limit(combined_context, max_tokens, model_name)

    #st.markdown("### Combined Text:")
    #st.write(combined_context)
    return combined_context

def original_context(question, llm, retriever, selected_prp_methods):

    vector_search_function = st.session_state.vector_search_function

    original_retrieval_results = run_pipeline(
        question,
        llm,
        retriever,
        query_transformation=None,  # Use the original query as is
        query_constructor=None,
        vector_search=vector_search_function,
        selected_prp_methods=selected_prp_methods,
        prompting=None  # No prompting needed for this step
    )
    st.write(original_retrieval_results)
    return original_retrieval_results


def prp_reranking(retrieval_results):

    if isinstance(retrieval_results[0], Document):
        retrieval_results = [retrieval_results]

    def reciprocal_rank_fusion(results: list[list], k=60):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results
    
    reranked_results = reciprocal_rank_fusion(retrieval_results)

    flat_reranked_results = [doc_score_tuple[0] for doc_score_tuple in reranked_results]


    st.markdown("### Top Ranked Results:")
    st.write(flat_reranked_results)
    return flat_reranked_results


def baseline_prompt(combined_context, question, llm):
    prompt_template = """
    
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        """
    
    prompt_baseline = ChatPromptTemplate.from_template(prompt_template)
    #st.write(combined_context)

    # Now use the combined context to generate the final answer
    prompt_input = {
        "context": combined_context,
        "question": question
    }

    # Use the baseline prompt for generating the answer
    response_chain = prompt_baseline | llm | StrOutputParser()
    response = response_chain.invoke(prompt_input)
    st.markdown("### Prompt:")
    st.write(prompt_input)
    st.markdown("### Answer:")
    st.write(response)
    return response

def contextual_compression(documents, question):
    embeddings = OpenAIEmbeddings()

    # Initialize Text Splitter
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator=". ")

    # Initialize Redundant Filter
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # Initialize Relevant Filter
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76, k=30)

    # Create Compressor Pipeline
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    # Compress the documents
    compressed_documents = pipeline_compressor.compress_documents(documents=documents, query=question)
    
    st.markdown("### Compressed Docs")
    display_documents(compressed_documents) 
    return compressed_documents

def filter_top_documents(documents):
    top_n = st.session_state.get('filter_number', 5)  # Default to 5 if not set
    top_docs = documents[:top_n]

    st.markdown("### Filter Top Docs")
    display_documents(top_docs) 
    return top_docs

def flatten_documents(nested_documents):
    """Flatten a list of lists of Document objects into a flat list of Document objects."""
    flatten_documents = [doc for sublist in nested_documents for doc in sublist]
    #st.write(flatten_documents)
    return flatten_documents

def custom_prompt(combined_context, question, llm):
    custom_template = st.session_state.get('custom_prompt_template', 'Default template if not set')

    prompt_custom = ChatPromptTemplate.from_template(custom_template)

    prompt_input = {
        "context": combined_context,
        "question": question
    }

    response_chain = prompt_custom | llm | StrOutputParser()
    response = response_chain.invoke(prompt_input)

    st.write("Question:", question)
    st.write("Answer:", response)
    return response

def run_baseline_pipeline(question, prompt_baseline, llm, retriever):

    # Retrieve context using the original query
    baseline_retriever = retriever.run(question)

    prompt_input = {
        "context": baseline_retriever,
        "question": question
    }

    # Read and generate the answer
    chain_baseline = (
        prompt_baseline
        | llm
        | StrOutputParser()
    )

    baseline_response = chain_baseline.invoke(prompt_input)
    st.markdown("### Original Question")
    st.write(question)

    st.markdown("### Retrieved Context for Baseline")
    st.write(baseline_retriever)

    st.markdown("### Baseline Model Response")
    st.write(baseline_response)

    return baseline_response


def step_back_prompt(combined_context, question, llm):
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
    step_back_chain = step_back_prompt | llm | StrOutputParser()
    response = step_back_chain.invoke(prompt_input)

    st.markdown("### Prompt:")
    st.write(prompt_input)
    st.markdown("### Answer:")
    st.write(response)

    return response

def run_step_back_pipeline (question, prompt_step_back, llm, retriever):
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

    step_back_chain = step_back_prompt_template | llm | StrOutputParser()
    step_back_query = step_back_chain.invoke({"question": question})

    # Retrieve context for both original and step-back questions
    normal_context = retriever.run(question)
    step_back_context = retriever.run(step_back_query)

    # Combine the contexts and form the final input for the answer generation
    prompt_input = {
        "normal_context": normal_context,
        "step_back_context": step_back_context,
        "question": question,  # Original question for generating the answer
    }
    
    # Generate the final answer using the combined context
    response_chain = prompt_step_back | llm | StrOutputParser()
    response = response_chain.invoke(prompt_input)

    st.markdown("### Original Question")
    st.write(question)

    st.markdown("### Step-Back Question")
    st.write(step_back_query)

    st.markdown("### Retrieved Context for Original Question")
    st.write(normal_context)

    st.markdown("### Retrieved Context for Step-Back Question")
    st.write(step_back_context)

    st.markdown("### Step-Back Prompting Response")
    st.write(response)

    return response

def run_mrq_fusion_pipeline (question, prompt_baseline, llm, retriever):
    # Define the prompt template for generating multiple queries
    mrq_prompt_template = """
    You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
    
    Provide these alternative questions separated by newlines.
    Original question: {question}
    """


    mrq_prompt = ChatPromptTemplate.from_template(mrq_prompt_template)


    # Define the parser to split the LLM result into a list of queries
    def mrq_parse_queries(text):
        return text.strip().split("\n")

    # Chain the prompt with the LLM and the parser
    mrq_chain = mrq_prompt | llm | StrOutputParser() | mrq_parse_queries

    # Generate the multiple queries from the original question
    generated_queries = mrq_chain.invoke({"question": question})

    #retriever_chains = {query: database_tool_instance.get_base_retriever() for query in generated_queries}

    #parallel_retriever = RunnableMap(steps=retriever_chains)
    #retrieval_results = parallel_retriever.invoke(generated_queries)

    base_retriever = retriever.get_base_retriever()
    retriever_map = base_retriever.map()

    #st.write(retriever_map)

    # Invoke the retriever_chain with generated_queries to get the documents
    retrieval_results = retriever_map.invoke(generated_queries)

    st.write(retrieval_results)

    # Apply reciprocal rank fusion to rerank the results
    reranked_results = reciprocal_rank_fusion(retrieval_results)

    top_reranked_results = reranked_results[:5]
    st.write(top_reranked_results)

    flat_reranked_results = [doc_score_tuple[0] for doc_score_tuple in top_reranked_results]

    embeddings = OpenAIEmbeddings()

    # Assuming you have the top 5 reranked documents in flat_reranked_results
    documents = flat_reranked_results

    # Initialize Text Splitter
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator=". ")

    # Initialize Redundant Filter
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # Initialize Relevant Filter
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76, k=30)

    # Create Compressor Pipeline
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    # Compress the documents
    compressed_documents = pipeline_compressor.compress_documents(
        documents=documents, query=question
    )
    st.write(compressed_documents)

    # Combine the page_content of each Document into a single string
    combined_context = "\n".join([doc.page_content for doc in compressed_documents])
    #st.write(combined_context)

    # Now use the combined context to generate the final answer
    prompt_input = {
        "context": combined_context,
        "question": question
    }

    # Use the baseline prompt for generating the answer
    response_chain = prompt_baseline | llm | StrOutputParser()
    response = response_chain.invoke(prompt_input)

    # Display the information in the UI
    st.markdown("### Original Question")
    st.write(question)
    st.markdown("### Generated Queries for MRQ")
    st.write(generated_queries)
    st.markdown("### Combined Retrieved Context")
    st.write(compressed_documents)
    st.markdown("### MRQ Model Response")
    st.write(response)

    return response

def run_pipeline(question, llm, query_transformation, query_constructor, vector_search, selected_prp_methods, prompting):
    # If a query transformation method is provided, use it to transform the question
    if query_transformation:
        transformed_questions = query_transformation(question)
    else:
        transformed_questions = question  # If none, just use the original question

    if query_constructor:
        constructed_retriever = query_constructor()
    else:
        constructed_retriever = None

    # Vector search with the transformed questions
    if vector_search:
        retrieval_results = vector_search(transformed_questions, constructed_retriever)

        if not retrieval_results:
            return st.warning("No retrieval results found.")

        if isinstance(retrieval_results[0], list):  # Flatten only if it's a list of lists
            retrieval_results = flatten_documents(retrieval_results)
    else:
        retrieval_results = []  # If none, there's no search results
        return st.warning("No retrieval results found.")

    # Apply selected PRP methods in order
    for method in selected_prp_methods:
        if method == 'Reranking':
            retrieval_results = prp_reranking(retrieval_results)
        elif method == 'Contextual Compression':
            retrieval_results = contextual_compression(retrieval_results, question)
        elif method == 'Filter Top Results':
            retrieval_results = filter_top_documents(retrieval_results)

    # Combine the page_content of each Document into a single string
    combined_context = create_combined_context(retrieval_results)

    if query_transformation == 'Step-Back Prompting':
        original_retrieval_results = original_context(question, llm, retriever, selected_prp_methods)
        combined_context += "\n" + original_retrieval_results
        st.write(combined_context)

    # Final prompting
    if prompting is None:
        return combined_context
    elif prompting:
        # Use the provided prompting method
        answer = prompting(combined_context, question, llm)
    else:
        # Fallback case
        answer = "No prompting method provided."

    return answer

def generate_metadata_descriptions(field_name, header_info_list, llm):
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
    chain = metadata_description_prompt_template | llm | StrOutputParser()

    response = chain.invoke(prompt_input)

    return response

def build_metadata_field_info(schema, header_info, llm):
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
        desc = generate_metadata_descriptions(field["name"], header_info, llm) + base_filter_instructions

        if "Header 3" in field["name"]:
            desc += header3_filter_instructions
        elif "Header 2" in field["name"]:
            desc += header2_filter_instructions
        elif "Header 1" in field["name"]:
            desc += header1_filter_instructions

        # Create the AttributeInfo object
        attr_info = AttributeInfo(
            name=field["name"],
            description=desc,
            type="string"
        )
        attr_info_list.append(attr_info)

    return attr_info_list

def document_loader_langchain(uploaded_file):
    documents = []
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        file_path = tmpfile.name
    
    try:
        file_name = os.path.splitext(uploaded_file.name)[0]
        file_type = "markdown" if file_path.endswith('.md') else "pdf"

        # Check file type and process accordingly
        if file_path.endswith('.md'):
            loader = TextLoader(file_path)
            documents = loader.load()

        elif file_path.endswith('.pdf'):
            # If PDF, use a PDF loader
            with open(file_path, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                # Load and process PDF file (not fully implemented)
                documents = [page.extract_text() for page in pdf_reader.pages]
        else:
            raise ValueError("Unsupported file format. Please upload a .md or .pdf file.")
        
        for doc in documents:
            doc.metadata.update({"file_name": file_name, "file_type": file_type})
    
    finally:
        # Ensure that the temporary file is cleaned up
        os.remove(file_path)

    st.markdown("### Loaded Documents:")
    st.write(documents)
    return documents

def document_loader_unstructured(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        file_path = tmpfile.name

    uploaded_document = partition_md(filename=file_path)

    os.remove(file_path)

    st.write(uploaded_document)
    return uploaded_document

def document_extractor_metadata(documents):
    if documents:
        # Extract metadata from the first document as representative
        doc_metadata = {
            "file_name": documents[0].metadata.get("file_name", "unknown"),
            "file_type": documents[0].metadata.get("file_type", "unknown")
        }
    else:
        doc_metadata = {"file_name": "unknown", "file_type": "unknown"}

    st.markdown("### Extracted Document Metadata:")
    st.write(doc_metadata)
    return doc_metadata

def document_splitter(documents):
    # Initialize a list to store the final chunks
    document_chunks = []
    
    # Recursive splitting logic
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.chunk_size, 
        chunk_overlap=st.session_state.chunk_overlap, 
        separators=st.session_state.selected_separators
    )

    # Perform splitting based on headers if checkbox is checked
    if st.session_state.split_by_headers:
        # Access headers to split on from session state
        headers_to_split_on = st.session_state.headers_to_split_on
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Perform header splitting and then recursive splitting on each header section
        for doc in documents:
            header_documents = markdown_splitter.split_text(doc.page_content)
            #st.write(header_documents)

            section_documents = text_splitter.split_documents(header_documents)

            document_chunks.extend(section_documents)
    else:
        # Perform recursive splitting directly if header splitting is not selected
        section_documents = text_splitter.split_documents(documents)
        document_chunks.extend(section_documents)

    st.markdown("### Splitted Document Chunks:")
    st.write(document_chunks)
    return document_chunks

def clean_chunks_content(document_chunks):
    cleaning_functions = [
        clean_extra_whitespace, 
        clean_dashes,
        clean_bullets, 
        clean_trailing_punctuation, 
        clean_ordered_bullets
    ]

    for document in document_chunks:
        cleaned_text = document.page_content
        for cleaning_function in cleaning_functions:
            # Apply each cleaning function to the text
            cleaned_text = cleaning_function(cleaned_text)
        
        # Update the document's page_content with the cleaned text
        document.page_content = cleaned_text

def extract_chunks_metadata(document_chunks):
    chunk_metadata_list = []
    for chunk in document_chunks:
        # Use a tuple to store both chunk and its metadata together
        metadata = chunk.metadata
        chunk_metadata_list.append((chunk.page_content, metadata))

    st.markdown("### Chunks Metadata:")
    st.write(chunk_metadata_list)
    return chunk_metadata_list

def build_index_schema(documents):
    schema = {"text": []}
    
    # Create a dictionary to store headers and their corresponding levels
    header_levels = {}

    # Iterate through each document to find unique headers and determine their levels
    for doc in documents:
        for header, _ in doc.metadata.items():
            level = header.count("#")  # Count number of '#' to determine the level
            header_levels[header] = level

    # Sort headers based on their levels (lower number = higher level)
    sorted_headers = sorted(header_levels.keys(), key=lambda h: header_levels[h])

    # Add sorted headers to the schema
    for header in sorted_headers:
        schema["text"].append({"name": header})

    # Convert schema to JSON and save to file
    with open(os.path.join("json", "index_schema.json"), "w") as file:
        json.dump(schema, file, indent=4)

    return schema

def create_final_documents(document_chunks, document_metadata):
    final_documents = []
    ids = []

    #build_index_schema(document_chunks)

    for chunk in document_chunks:
        # Generate a unique ID for each document chunk
        doc_id = f"{document_metadata['file_name']}_{uuid.uuid4()}"
        ids.append(doc_id)

        chunk_metadata = chunk.metadata
        # Merge the chunk-level metadata with the document-level metadata
        merged_metadata = {"unique_id": doc_id, **document_metadata, **chunk_metadata}
        
        final_doc = Document(page_content=chunk.page_content, metadata=merged_metadata)
        final_documents.append(final_doc)

    st.markdown("### Processed Document Chunks:")
    st.write(final_documents)
    return final_documents

def build_chroma_vectorstore(document_chunks, embeddings):
    chroma_client = st.session_state.client_db.client
    chroma_collection_name = st.session_state.collection_name
    
    # Use the ids that were passed to the function, which already include the file name
    ids = [doc.metadata["unique_id"] for doc in document_chunks]
    vectorstore = Chroma.from_documents(
        documents=document_chunks, 
        embedding=embeddings, 
        ids=ids,  # This list is now passed in as a parameter
        client=chroma_client, 
        collection_name=chroma_collection_name
    )

    st.markdown("### Build Vectorstore:")
    st.write(vectorstore)
    return vectorstore

def build_redis_vectorstore(document_chunks, embeddings):
    redis_url = st.session_state.redis_url
    index_schema = build_index_schema(document_chunks)
    
    redis_vectorstore = Redis.from_documents(
        document_chunks,
        embeddings,
        redis_url=redis_url,
        index_name = "user",
        index_schema=index_schema
    )

    return redis_vectorstore

def run_ingestion_pipeline(uploaded_file, document_loader, document_embedder, vectorstore_method):
    if uploaded_file is not None:
        loaded_documents = document_loader(uploaded_file)
        document_metadata = document_extractor_metadata(loaded_documents)
        document_chunks = document_splitter(loaded_documents)
        clean_chunks_content(document_chunks)

        processed_doc_chunks = create_final_documents(document_chunks, document_metadata)

        if vectorstore_method:
            vectorstore = vectorstore_method(processed_doc_chunks, document_embedder)
            st.success("Ingestion Complete")
        else:
            raise ValueError("Selected vectorstore type is not supported yet.")

        return vectorstore
    else:
        raise ValueError("No file uploaded. Please upload a file to proceed with indexing.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Playground", page_icon="🕹️", layout="wide")
    apply_css()
    st.title("PLAYGROUND 🕹️")
    
    with st.empty():
        Init.initialize_session_state()
        Init.initialize_agent_state()
        Init.initialize_clientdb_state()
        if 'query_transformer' not in st.session_state:
            st.session_state.query_transformer = QueryTransformer()
        if 'query_constructor' not in st.session_state:
            st.session_state.query_constructor = QueryConstructor()
        if 'vector_search_instance' not in st.session_state:
            st.session_state.vector_search_instance = VectorSearch()

    with st.sidebar:
            existing_collections = st.session_state.client_db.get_existing_collections()
            if not existing_collections:
                st.warning("No collections available.")
            else:
                selected_collection_name, selected_collection_object = Main.handle_collection_selection(existing_collections)
                st.session_state.collection_name = selected_collection_name
            
    Sidebar.file_upload_and_ingest(st.session_state.client_db, selected_collection_name, selected_collection_object)

    with st.expander("View RAG Pipeline Diagram"):
        st.image("image/RAGAv2.png", caption="Retrieval-Augmented Generation (RAG) Pipeline Diagram")
        

    indexing_tab, rag_tab = st.tabs(["Indexing", "Augmented Generation"])

    with indexing_tab:
        st.header("Indexing Setup for RAG Pipeline")
    
        #DOCUMENT LOADER
        with st.container():
            document_loader_methods = {
                'LangChain': document_loader_langchain,
                'Unstructured': document_loader_unstructured,
            }
            document_loader_selected = st.selectbox(
                "Choose a document loader method",
                options=list(document_loader_methods.keys()),
                index=0  # Default to 'Unstructured'
            )

            uploaded_file = st.file_uploader("Upload your document (Markdown or PDF)", type=['md', 'pdf'])

        #SPLITTER
        with st.container():
            # Selectbox for choosing the text splitter type
            splitter_type = st.selectbox(
                "Select the type of text splitter",
                options=["None", "Recursive Splitter"],
                index=0  # Default to "None"
            )

            if splitter_type == "Recursive Splitter":
                with st.expander("TS Setting"):
                    chunk_size = st.slider(
                        "Choose the size of text chunks",
                        min_value=100, max_value=5000, value=3000, step=100
                    )
                    st.session_state.chunk_size = chunk_size
                    
                    chunk_overlap = st.slider(
                        "Choose the overlap between text chunks",
                        min_value=0, max_value=500, value=200, step=10
                    )
                    st.session_state.chunk_overlap = chunk_overlap

                separator_display_mapping = {
                    "\\n\\n (Double Newline)": "\n\n",
                    "\\n (Newline)": "\n",
                    "Space": " ",
                    "Empty String": ""
                }

                # At the start of your Streamlit app, initialize selected_separators
                if 'selected_separators' not in st.session_state:
                    st.session_state.selected_separators = list(separator_display_mapping.values())

                # Then, in your UI logic, update selected_separators based on user interaction
                separator_options = list(separator_display_mapping.keys()) + ["ALL"]
                default_selections = list(separator_display_mapping.keys())  # Default to all separators selected

                selected_separators_display = st.multiselect(
                    "Choose the separators for splitting",
                    options=separator_options,
                    default=default_selections
                )

                # If "ALL" is selected, use all separators, otherwise only selected ones
                if "ALL" in selected_separators_display:
                    st.session_state.selected_separators = list(separator_display_mapping.values())
                else:
                    st.session_state.selected_separators = [
                        separator_display_mapping[disp] 
                        for disp in selected_separators_display 
                        if disp in separator_display_mapping
                    ]

                ###
                default_headers = [("##", "h1_main"), ("###", "h2_chapter"), ("####", "h3_subchapter")]

                if 'split_by_headers' not in st.session_state:
                    st.session_state.split_by_headers = False

                if 'headers_to_split_on' not in st.session_state or st.session_state.split_by_headers:
                    st.session_state.headers_to_split_on = default_headers.copy()

                def toggle_split_by_headers():
                    st.session_state.split_by_headers = not st.session_state.split_by_headers
                    if st.session_state.split_by_headers:
                        st.session_state.headers_to_split_on = default_headers.copy()
                    else:
                        st.session_state.headers_to_split_on = []

                # Checkbox for splitting by headers
                split_by_headers = st.checkbox(
                    "Split by headers",
                    value=st.session_state.get('split_by_headers', False),
                    on_change=toggle_split_by_headers
                )

                # Function to add a new header
                def add_header():
                    next_header_index = len(st.session_state.headers_to_split_on) + 1
                    next_header_sign = "#" * next_header_index
                    next_header_name = f"Header {next_header_index}"
                    st.session_state.headers_to_split_on.append((next_header_sign, next_header_name))

                # Function to remove the last header
                def remove_header():
                    if st.session_state.headers_to_split_on:
                        st.session_state.headers_to_split_on.pop()
                
                if split_by_headers:
                    # Display current headers and allow editing
                    for index, (header_sign, header_name) in enumerate(st.session_state.headers_to_split_on):
                        with st.container():
                            cols = st.columns([1, 4])
                            with cols[0]:
                                new_header_sign = st.text_input(f"Header sign {index+1}", value=header_sign)
                            with cols[1]:
                                new_header_name = st.text_input(f"Header name {index+1}", value=header_name)
                            if new_header_sign != header_sign or new_header_name != header_name:
                                st.session_state.headers_to_split_on[index] = (new_header_sign, new_header_name)


                    # Buttons to add/remove headers
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("Add header", on_click=add_header)
                    with col2:
                        st.button("Remove header", on_click=remove_header)

                    # Show the headers as they will be used in the splitter
                    st.write("Headers to split on:")
                    #for header in st.session_state.headers_to_split_on:
                        #st.write(f"{header[0]}: {header[1]}")
                    st.write(st.session_state.headers_to_split_on)
        
        #EMBEDDER:
        with st.container():
            embedding_models = {
                'OpenAI Embedding': OpenAIEmbeddings()
            }


            embedding_model_selected = st.selectbox(
                "Select the embedding model",
                options=list(embedding_models.keys()),
                index=0  # Default to OpenAI Embedding
            )

            st.session_state.document_embedder = embedding_models[embedding_model_selected]

        #VECTORSTORE
        with st.container():
            vectorstore_methods = {
                'Chroma': build_chroma_vectorstore,
                'Redis': build_redis_vectorstore,
                'Other': None  # Placeholder for other types
            }

            vectorstore_type = st.selectbox(
                "Select the vectorstore type",
                options=list(vectorstore_methods.keys()),  # Dynamically list available options
                index=0  # Default selection can be Chroma or Redis depending on preference
            )

            if vectorstore_type == 'Redis':
                st.session_state.redis_url = 'redis://localhost:9000'  # Default Redis URL
                st.text_input("Redis URL", value=st.session_state.redis_url, disabled=True)
                index_name = "user"  # Replace with actual index name or user input

                def load_index_schema(schema_file_path):
                    with open(schema_file_path, 'r') as file:
                        schema = json.load(file)
                    return schema
                
                index_schema_file_path = os.path.join("json", "index_schema.json")
                index_schema = load_index_schema(index_schema_file_path)
                st.session_state.index_schema = index_schema
                
                # Initializing the Redis vectorstore
                st.session_state.vector_store = Redis(
                    redis_url=st.session_state.redis_url,
                    index_name=index_name,
                    embedding=st.session_state.document_embedder,
                    index_schema=index_schema,  # You can define the index schema or leave it as None
                )

        if st.button("Ingest Document"):
            ingestion = run_ingestion_pipeline(
                uploaded_file, 
                document_loader=document_loader_methods[document_loader_selected],
                document_embedder=embedding_models[embedding_model_selected],
                vectorstore_method=vectorstore_methods[vectorstore_type],
                )



    with rag_tab:
    #QUERY TRANSFORMATION
        with st.container():

            query_transformer = st.session_state.query_transformer

            query_transformation_methods = {
                'None': None, 
                'Multi-Retrieval Query': query_transformer.multi_retrieval_query,
                'Rewrite-Retrieve-Read': query_transformer.rewrite_retrieve_read,
                'Query Extractor': query_transformer.query_extractor,
                'Step-Back Prompting': query_transformer.step_back_prompting,
            }

            selected_query_transformation = st.selectbox(
                "Select the query transformation method:",
                options=list(query_transformation_methods.keys())
            )

            qt_expander_title = f"{selected_query_transformation} Explanation" if selected_query_transformation != 'None' else "Query Transformation Example"
            with st.expander(qt_expander_title):
                if selected_query_transformation == 'Multi-Retrieval Query':
                    st.markdown(
                        '''
                        ### Explanation 
                        
                        This method generates multiple variations of the original question to enhance the retrieval process by covering different phrasings or aspects of the query.

                        ### Example

                        **Input**: Original Question
                        ```
                        "How to deposit a cheque issued to an associate in my business into my business account?"
                        ```

                        **Output**: Multi Generated Questions
                        ```
                        1. "What is the process for depositing a cheque made out to an associate into my business account?"
                        2. "Can you explain how to deposit a cheque issued to an associate into my business account?"
                        3. "What are the steps to deposit a cheque made out to an associate into my business account?"
                        ```
                        '''
                    )
                elif selected_query_transformation == 'Rewrite-Retrieve-Read':
                    st.markdown(
                        '''
                        ### Explanation 
                        The Rewrite-Retrieve-Read method improves the retrieval process by first prompting an LLM to generate a more precise and focused search query from a user's initial vague or poorly structured question. This enhanced query is then used to retrieve relevant documents, which are analyzed by the LLM to formulate a comprehensive response.

                        ### Example
                        **Input**: Original Question
                        ```
                        "I wanna know about the stuff that makes computers think?"
                        ```
                        
                        **Output**: Rewritten Question
                        ```
                        "Key principles and technologies in artificial intelligence and machine learning for computer cognition"
                        ```
                        '''
                    )

                elif selected_query_transformation == 'Query Extractor':
                    st.markdown(
                        '''
                        ### Explanation
                        
                        The Query Extractor refines the user's natural language query to improve the retrieval of relevant documents from a vector database.
                        
                        ### Example
                        
                        **Input**: Original Question
                        ```
                        "What are the main factors contributing to urban pollution?"
                        ```
                        
                        **Output**: Extracted Question
                        ```
                        "urban pollution main contributing factors"
                        ```
                        '''
                    )


            if selected_query_transformation == 'Multi-Retrieval Query':
                with st.expander("Multi-Retrieval Query Settings"):
                    query_count = st.slider("Select the number of queries to generate", 1, 10, 3, 1)
                    st.session_state.query_count = query_count

                    default_mrq_prompt = """
                    You are an AI language model assistant. Your task is to generate {query_count} different versions of the given user question to retrieve relevant documents from a vector database.
                    
                    Provide these alternative questions separated by newlines.
                    Original question: {question}
                    """
                    mrq_prompt_template = st.text_area("Customize your Multi-Retrieval Query prompt:", value=default_mrq_prompt, height=200)
                    st.session_state.mrq_prompt_template = mrq_prompt_template

            if selected_query_transformation == 'Step-Back Prompting':
                st.session_state.selected_prompting = 'Step-Back Prompt'

                vector_search_original_query = st.checkbox(
                "Perform vector search on both original and step-back queries", value=True
            )

    #QUERY CONSTRUCTION     
        with st.container():

            query_constructor = st.session_state.query_constructor 

            query_constructor_methods = {
                'None': None, 
                'Self-Query Construction': query_constructor.self_query_constructor,
            }

            def on_query_constructor_change():
                if st.session_state.query_constructor_selection == 'Self-Query Construction':
                    st.session_state.vector_search_selection = 'Self-Query Retrieval'
                else:
                    st.session_state.vector_search_selection = 'None'

            selected_query_constructor = st.selectbox(
                "Select the query constructor method:",
                options=list(query_constructor_methods.keys()),
                on_change=on_query_constructor_change,
                key='query_constructor_selection'
            )

            qc_expander_title = f"{selected_query_constructor} Explanation" if selected_query_constructor != 'None' else "Query Constructor Example"
            with st.expander(qc_expander_title):
                if selected_query_constructor == 'Self-Query Retrieval':
                    st.markdown(
                        '''
                        **SelfQuery Retrieval:**
                        
                        This method translates natural language queries into structured queries using metadata filtering. This is particularly effective when dealing with vector databases that include structured data.
                        
                        **Example:**
                        
                        Given a natural language query like "What are movies about aliens in the year 1980", SelfQuery Retrieval will decompose the query into a semantic search term and logical conditions for metadata filtering, thus leveraging both semantic and structured data search capabilities.
                        '''
                    )
            
            with st.expander("QC Settings:"):
                if selected_query_constructor == 'Self-Query Construction':

                    metadata_info_file_path = 'json/metadata_field_info.json'
                    metadata_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106", verbose=True)
                
                    if 'metadata_field_info' not in st.session_state:
                        if os.path.exists(metadata_info_file_path):
                            # Load existing metadata field info
                            with open(metadata_info_file_path, 'r') as file:
                                st.session_state.metadata_field_info = json.load(file)
                        else:
                            # Generate and save new metadata field info
                            json_file_path = 'json/index_schema.json'
                            with open(json_file_path, 'r') as file:
                                index_schema = json.load(file)
                            st.session_state.metadata_field_info = build_metadata_field_info(index_schema, st.session_state.headers_to_split_on, metadata_llm)
                            
                            # Save the generated metadata field info
                            with open(metadata_info_file_path, 'w') as file:
                                json.dump([attr_info.__dict__ for attr_info in st.session_state.metadata_field_info], file, indent=4)

                    if 'document_content_description' not in st.session_state:
                        st.session_state.document_content_description = "Structured sections of the Danish Building Regulation 2018 document"

                        # Function to add a new attribute
                    def add_attribute():
                        next_index = len(st.session_state.metadata_field_info) + 1
                        st.session_state.metadata_field_info.append(
                            AttributeInfo(name=f"Header{next_index}", description=f"Header {next_index} Content", type="string")
                        )

                        # Function to remove the last attribute
                    def remove_attribute():
                        if st.session_state.metadata_field_info:
                            st.session_state.metadata_field_info.pop()

                        # Display current attributes and allow editing
                    for index, attribute_info in enumerate(st.session_state.metadata_field_info):
                        with st.container():
                            pass
                        # Buttons to add/remove attributes
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("Add attribute", on_click=add_attribute, key="add_attribute")
                    with col2:
                        st.button("Remove attribute", on_click=remove_attribute, key="remove_attribute")

                    # Input for document content description
                    st.session_state.document_content_description = st.text_area(
                        "Document Content Description",
                        value=st.session_state.document_content_description
                        )

                    # Optionally display the current configuration
                    # st.write("Current Metadata Fields:")
                    # for attribute_info in st.session_state.metadata_field_info:
                    #     st.write(f"Name: {attribute_info.name}, Description: {attribute_info.description}")

    #VECTOR SEARCH
        with st.container():

            vector_search_instance = st.session_state.vector_search_instance

            vector_search_methods = {
                'None': None, 
                'Base Retriever': vector_search_instance.base_retriever,
                'Reranking Retriever': vector_search_instance.reranking_retriever,
                'Self-Query Retrieval': vector_search_instance.self_query_retriever,
            }
            
            default_vector_search = st.session_state.get('vector_search_selection', 'None')

            selected_vector_search = st.selectbox(
                "Select the vector/similarity search method:",
                options=list(vector_search_methods.keys()),
                index=list(vector_search_methods.keys()).index(default_vector_search),
                key='vector_search_selection'
            )

            st.session_state.vector_search_function = vector_search_methods[selected_vector_search]

            vs_expander_title = f"{selected_vector_search} Explanation" if selected_vector_search != 'None' else "Vector Search Example"
            with st.expander(vs_expander_title):
                all_documents_str = '''
                    "doc1": "Climate change and economic impact.",
                    "doc2": "Public health concerns due to climate change.",
                    "doc3": "Climate change: A social perspective.",
                    "doc4": "Technological solutions to climate change.",
                    "doc5": "Policy changes needed to combat climate change.",
                    "doc6": "Climate change and its impact on biodiversity.",
                    "doc7": "Climate change: The science and models.",
                    "doc8": "Global warming: A subset of climate change.",
                    "doc9": "How climate change affects daily weather.",
                    "doc10": "The history of climate change activism."
                    '''

                if selected_vector_search == 'Base Retriever':
                    st.markdown(
                        '''
                        ### Explanation 

                        The Base Retriever method retrieves documents based on similarity to the input query using vector embeddings.

                        ### Example

                        **Documents**:
                        ```
                        "doc1": "Climate change and economic impact.",
                        "doc2": "Public health concerns due to climate change.",
                        "doc3": "Climate change: A social perspective.",
                        "doc4": "Technological solutions to climate change.",
                        "doc5": "Policy changes needed to combat climate change.",
                        "doc6": "Climate change and its impact on biodiversity.",
                        "doc7": "Climate change: The science and models.",
                        "doc8": "Global warming: A subset of climate change.",
                        "doc9": "How climate change affects daily weather.",
                        "doc10": "The history of climate change activism."
                        ```

                        **Input:** Original Question  
                        
                        ```
                        "impact of climate change"
                        ```

                        **Output:** Retrieved Documents (top k = 3)
                        
                        ```
                        "doc6": "Climate change and its impact on biodiversity."
                        "doc1": "Climate change and economic impact."
                        "doc9": "How climate change affects daily weather."
                        ```
                        '''
                    )
                elif selected_vector_search == 'Reranking Retriever':
                    st.markdown(
                        '''
                        ### Explanation 
                        This method first retrieves a set of documents similar to the query and then reranks them based on additional criteria to improve relevance.

                        ### Example

                        **Documents**:
                        
                        ```
                        "doc1": "Climate change and economic impact.",
                        "doc2": "Public health concerns due to climate change.",
                        "doc3": "Climate change: A social perspective.",
                        "doc4": "Technological solutions to climate change.",
                        "doc5": "Policy changes needed to combat climate change.",
                        "doc6": "Climate change and its impact on biodiversity.",
                        "doc7": "Climate change: The science and models.",
                        "doc8": "Global warming: A subset of climate change.",
                        "doc9": "How climate change affects daily weather.",
                        "doc10": "The history of climate change activism."
                        ```

                        **Input:** Original Question  
                        ```
                        impact of climate change
                        ```

                        **Output:** Top Ranked Documents  

                        ```
                        1. "doc2": "Public health concerns due to climate change.", score: 0.066
                        2. "doc3": "Climate change: A social perspective.", score: 0.064
                        3. "doc9": "How climate change affects daily weather.", score: 0.048
                        4. "doc6": "Climate change and its impact on biodiversity.", score: 0.033
                        5. "doc1": "Climate change and economic impact.", score: 0.017
                        6. "doc4": "Technological solutions to climate change.", score: 0.017
                        7. "doc5": "Policy changes needed to combat climate change.", score: 0.016
                        ```
                        '''
                    )
                
            if selected_vector_search != 'None':
                with st.expander("Retriever Settings"):
                    top_k_value = st.slider("Select the top k value for retrieval", 1, 20, 3, 1)
                    st.session_state.top_k_value = top_k_value

                    search_type = st.selectbox("Select the search type", ["similarity", "mmr", "similarity_score_threshold"])
                    st.session_state.search_type = search_type

                    if search_type == "mmr":
                        lambda_mult = st.slider("Select the lambda multiplier for MMR", 0.0, 1.0, 0.5, 0.01)
                        st.session_state.lambda_mult = lambda_mult

                        fetch_k = st.number_input("Select the fetch k value for MMR", 1, 50, 20, 1)
                        st.session_state.fetch_k = fetch_k

                    if search_type == "similarity_score_threshold":
                        score_threshold = st.slider("Set the similarity score threshold", 0.0, 1.0, 0.8, 0.01)
                        st.session_state.score_threshold = score_threshold

            if selected_vector_search == 'Reranking Retriever':
                with st.expander("Reranking Retriever Settings"):
                    # Slider to set the number of top reranked results
                    top_reranked_value = st.slider("Select the number of top reranked results", 1, 20, 5, 1)
                    st.session_state.top_reranked_value = top_reranked_value

                    # Slider to set the k value for reciprocal rank fusion
                    reciprocal_rank_k = st.slider("Select the k value for reciprocal rank fusion", 1, 100, 60, 1)
                    st.session_state.reciprocal_rank_k = reciprocal_rank_k

    #POST-PROCESSING RETRIEVAL
        with st.container():
            prp_methods = ['Reranking', 'Contextual Compression', 'Filter Top Results']
            selected_prp_methods = st.multiselect("Select and order the Post-Retrieval Processing methods:", prp_methods)

            if 'Filter Top Results' in selected_prp_methods:
                st.session_state.filter_number = st.slider("Select the number of top results to filter", min_value=1, max_value=20, value=5, step=1)
            else:
                st.session_state.filter_number = 5 

            prp_expander_title = f"{selected_prp_methods} Explanation" if selected_prp_methods != 'None' else "Post-Retrieval Processing Example"
            with st.expander(prp_expander_title):
                if 'Reranking' in selected_prp_methods:
                    st.markdown(
                        '''
                        ### Reranking

                        **Explanation:**  
                        Reranking involves reordering the initially retrieved documents based on additional relevance criteria or scores.

                        **Example:**  
                        Given initial retrieval results, reranking might prioritize documents more closely related to specific aspects of the query, resulting in a more refined set of top documents.
                        '''
                    )

                if 'Contextual Compression' in selected_prp_methods:
                    st.markdown(
                        '''
                        ### Contextual Compression

                        **Explanation:**  
                        This method compresses the retrieved documents by removing redundant information and focusing on the most relevant content related to the query.

                        **Example:**  
                        From lengthy documents, only the sections or sentences that are most relevant to the query are retained, providing a concise and focused set of documents.
                        '''
                    )

                if 'Filter Top Results' in selected_prp_methods:
                    st.markdown(
                        '''
                        ### Filter Top Results
                        **Explanation:**  
                        This process filters the retrieved or reranked documents to only include the top N results, based on a specified number.

                        **Example:**  
                        If the filter number is set to 5, only the top 5 documents (based on relevance or reranking scores) are selected for further processing.
                        '''
                    )

    #PROMPTING
        with st.container():
            prompting_methods = {
                'None': None, 
                'Baseline Prompt': baseline_prompt, 
                'Custom Prompting': custom_prompt,
                'Step-Back Prompt': step_back_prompt
            }

            default_prompting_method = st.session_state.get('selected_prompting', 'Baseline Prompt')

            selected_prompting = st.selectbox(
                "Select the prompting method:",
                options=list(prompting_methods.keys()),
                index=list(prompting_methods.keys()).index(default_prompting_method)
            )

            if selected_prompting == 'Custom Prompting':
                user_defined_prompt = st.text_area("Enter your custom prompt template here:")
                full_custom_prompt = user_defined_prompt + "\n\nContext: {context}\n\nQuestion: {question}"
                st.session_state.user_custom_prompt = user_defined_prompt
                st.session_state.full_custom_prompt = full_custom_prompt

            prompt_expander_title = f"{selected_prompting} Explanation" if selected_prompting != 'None' else "Prompting Example"
            with st.expander(prompt_expander_title):
                if selected_prompting == 'Baseline Prompt':
                    st.markdown(
                        '''
                        ### Baseline Prompt
                        **Explanation:**  
                        Uses a standard template to frame the context and question for generating the response.

                        **Example:**  
                        ```
                        Use the following pieces of context to answer the question at the end. 
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.

                        Context: {context}

                        Question: {question}
                        ```
                        '''
                    )
                elif selected_prompting == 'Custom Prompting':
                    custom_example = st.session_state.get('user_custom_prompt', 'No custom prompt defined.')
                    st.markdown(
                        f"""
                        ### Custom Prompting

                        **Explanation:**  
                        Allows users to define their own prompt template, giving more control over how the AI model generates responses.

                        **Example:**  
                        
                        ```
                        {custom_example}

                        Context: {{context}}
                        
                        Question: {{question}}
                        ```
                        
                        """
                    )

        user_question = st.text_input("Type Something Here")

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106", verbose=True)
        retriever = DatabaseTool(
            llm=llm,
            vector_store=st.session_state.vector_store,
        )


        col1, col2 = st.columns(2)

        generate_answer = st.button("Generate")
        if generate_answer:
            with col1:
                st.subheader("Baseline Pipeline")
                
                #baseline_answer = run_pipeline(
                    #user_question,
                    #llm,
                    #retriever,
                    #query_transformation=None,
                    #vector_search=base_vector_search,
                    #reranking=False,
                    #compression=False,
                    #prompting=baseline_prompt
                #)

            with col2:
                st.subheader("Query Transformation Pipeline")
                st.markdown("### Query")
                st.write(user_question)

                custom_answer = run_pipeline(
                    user_question,
                    llm,
                    query_transformation=query_transformation_methods[selected_query_transformation],
                    query_constructor = query_constructor_methods[selected_query_constructor],
                    vector_search=vector_search_methods[selected_vector_search],
                    selected_prp_methods=selected_prp_methods,  # Passing the selected PRP methods
                    prompting=prompting_methods[selected_prompting]
                )

    #st.write(st.session_state.top_k_value)
    #st.write(st.session_state.search_type)
    #st.write(st.session_state.score_threshold)
    #st.write(st.session_state.selected_vector_search)
    st.write(st.session_state.client_db)
    st.write(st.session_state.vector_store)
    st.write(selected_collection_name)
    st.write(st.session_state.collection_name)
    #st.write(st.session_state.headers_to_split_on)
    st.write(st.session_state.selected_document)

if __name__ == "__main__":
    main()