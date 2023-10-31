import streamlit as st
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from ragas import evaluate
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import shutil
import os
from ragas.testset import TestsetGenerator
from agent.tools import DatabaseTool
from langchain.chat_models import ChatOpenAI
from utility.sessionstate import Init
from langchain.chains.question_answering import load_qa_chain
import openai
from datasets import Dataset
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser 
import ast
from utility.ingestion import PDFTextExtractor


LOG_FILE = "./evaluation/evaluation_logs.csv"

def save_to_directory(df, test_generation_name):
    """Save the dataframe to a specific directory as a CSV file."""
    
    # Hardcoded directory and file name
    root_directory = "evaluation"
    folder_name = "dataset"
    file_name = f"{test_generation_name}.csv"

    # Check and create the directory structure if it doesn't exist
    directory_path = os.path.join(root_directory, folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Save the dataframe as a CSV file
    file_path = os.path.join(directory_path, file_name)
    df.to_csv(file_path, index=False)

    return file_path

def load_evaluation_logs():
    if os.path.exists(LOG_FILE):
        # Check if the file is empty
        if os.path.getsize(LOG_FILE) > 0:
            return pd.read_csv(LOG_FILE).to_dict(orient='records')
        else:
            return []
    else:
        return []

def save_evaluation_logs(logs):
    df = pd.DataFrame(logs)
    df.to_csv(LOG_FILE, index=False)

def save_processed_df(df, filename):
    folder_path = os.path.dirname(filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df.to_csv(filename, index=False)

def save_eval_dataframe(df, filename, log_type, timestamp, update_log=True):
    folder_path = f"./evaluation/{timestamp}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, filename)
    df.to_csv(filepath, index=False)

    if update_log:
        # Update the master log file
        log_entry = {"timestamp": timestamp, "log_type": log_type, "file_path": filepath}

        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            master_log = pd.read_csv(LOG_FILE)
        else:
            master_log = pd.DataFrame(columns=["timestamp", "log_type", "file_path"])
        
        # The line that is causing the issue
        new_entry_df = pd.DataFrame([log_entry])
        master_log = pd.concat([master_log, new_entry_df], ignore_index=True)
        master_log.to_csv(LOG_FILE, index=False)
    
    return filepath

def run_evaluation(dataset: Dataset, evaluation_quantity: int):

    result = evaluate(
            dataset.select(range(evaluation_quantity)),  # Limiting to 3 for example
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )
    return result

@st.cache_data()
def prepare_dataset():
    # Load the existing dataset
    fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")

    # Remove the 'answer' and 'contexts' columns
    fiqa_eval = fiqa_eval.remove_columns(['answer', 'contexts'])

    # Convert the modified dataset to a pandas DataFrame
    fiqa_eval_df = fiqa_eval["baseline"].to_pandas()

    return fiqa_eval_df

def populate_dataset_with_context_and_answer(df, num_rows=None):
    batch_size = st.session_state.batch_size
    # Ensure not to exceed the actual number of rows in the dataframe
    num_rows = min(num_rows, len(df))

    # Slice the dataframe to only take the desired number of rows
    df = df.iloc[:num_rows]

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", verbose=True)

    template = """
        {system_message_content}
        {formatting_message_content}

        Context:
        {retrieved_context}
        Question:
        {question}
    """

    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm | StrOutputParser()
    
    df_batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
    
    # Initialize empty lists to store generated contexts and answers
    generated_contexts = []
    generated_answers = []

    db_tool_instance = DatabaseTool(
        llm=llm,
        vector_store=st.session_state.vector_store,
    )

    # Loop through each row in the DataFrame
    for batch in df_batches:
        prompts_data = []
        
        # Prepare the input data for each question in the batch
        for _, row in batch.iterrows():
            question = row['question']
            retrieved_context = db_tool_instance.run(question)

            prompts_data.append({
                "system_message_content": st.session_state.system_message_content,
                "formatting_message_content": st.session_state.formatting_message_content,
                "retrieved_context": retrieved_context,
                "question": question
            })

        # Generate answers using the chain's batch method
        responses = chain.batch(prompts_data, config={"max_concurrency": batch_size})

        # Extract the generated contexts and answers from the responses
        for data, response in zip(prompts_data, responses):
            generated_contexts.append([data['retrieved_context']])
            generated_answers.append(response)

    # Create new columns in the DataFrame and populate them with the generated contexts and answers
    df['contexts'] = generated_contexts
    df['answer'] = generated_answers
    
    return df

def batchify(data, batch_size):
    """Utility function to split data into batches"""
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

def main():
    load_dotenv()
    st.set_page_config(page_title="EVALUATION", page_icon="", layout="wide")
    st.title("RAG Evaluation")

    #st.write(evaluation_logs)

    with st.empty():
        Init.initialize_session_state()
        if 'df_detailed_display' not in st.session_state:
            st.session_state.df_detailed_display = None
        if 'df_summary_display' not in st.session_state:
            st.session_state.df_summary_display = None
        if 'evaluation_quantity' not in st.session_state:
            st.session_state.evaluation_quantity = 5  # Default value
        if 'test_size' not in st.session_state:
            st.session_state.test_size = 10
        evaluation_logs = load_evaluation_logs()
    
    with st.sidebar:
        st.title("Settings")

        evaluation_name = st.text_input("Name of the Evaluation")
        if not evaluation_name.strip():  # Check if the evaluation_name is empty or just whitespace
            evaluation_name = "Unnamed_Evaluation"
        st.session_state.evaluation_name = evaluation_name

        evaluation_quantity = st.sidebar.slider(
            "Evaluation Range", 
            min_value=1, 
            max_value=30, 
            value=st.session_state.evaluation_quantity, 
            step=1)
        
        st.session_state.evaluation_quantity = evaluation_quantity

        batch_size = st.sidebar.slider(
            "Batch Size", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.get("batch_size", 3),  # Default to 3 if not previously set
            step=1
        )
        st.session_state.batch_size = batch_size

        system_message_content = st.text_area("System Message", st.session_state.system_message_content, height=200, max_chars=1500)

        run_evaluation_button = st.button("Run Evaluation")


    evaluation_tab, test_generator_tab = st.tabs(["Evaluation", "Test Data Generation"])

    with evaluation_tab:

        uploaded_file = st.file_uploader("Upload a Prepared Dataset (CSV)", type=["csv"])
        if uploaded_file:
            uploaded_df = pd.read_csv(uploaded_file)
            uploaded_df['contexts'] = uploaded_df['contexts'].apply(ast.literal_eval)
            uploaded_df['ground_truths'] = uploaded_df['ground_truths'].apply(ast.literal_eval)
            st.dataframe(uploaded_df)

        else:
            uploaded_df = None


        if run_evaluation_button:

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if uploaded_df is not None:
                processed_df = uploaded_df
            else:
                fiqa_eval_df = prepare_dataset()
                st.dataframe(fiqa_eval_df)

                processed_df = populate_dataset_with_context_and_answer(fiqa_eval_df, st.session_state.evaluation_quantity)
                st.dataframe(processed_df)

                processed_filename = f"{evaluation_name} - processed - {timestamp}.csv"
                processed_filepath = save_eval_dataframe(processed_df, processed_filename, "processed", timestamp, update_log=False)

            processed_dataset = Dataset.from_pandas(processed_df)
            result = run_evaluation(processed_dataset, st.session_state.evaluation_quantity)

            
            df_summary = pd.DataFrame.from_dict(result, orient='index', columns=['Value']).reset_index()
            df_summary.columns = ['Metric', 'Value']
            st.session_state.df_summary_display = df_summary
            summary_filename = f"{evaluation_name} - summary - {timestamp}.csv"
            summary_filepath = save_eval_dataframe(df_summary, summary_filename, "summary", timestamp)

            # Save and display detailed DataFrame
            df_detailed = result.to_pandas()
            st.session_state.df_detailed_display = df_detailed
            detailed_filename = f"{evaluation_name} - detailed - {timestamp}.csv"
            detailed_filepath = save_eval_dataframe(df_detailed, detailed_filename, "detailed", timestamp)
            
            log_entry = {
                "timestamp": timestamp,
                "DataFrame_Summary": summary_filepath,
                "DataFrame_Detailed": detailed_filepath,
                **result
            }
            evaluation_logs.append(log_entry)
            save_evaluation_logs(evaluation_logs)

            st.experimental_rerun()


        if st.session_state.df_summary_display is not None:
            st.subheader("Summary")
            st.dataframe(st.session_state.df_summary_display, hide_index=True)

        if st.session_state.df_detailed_display is not None:
            st.subheader("Detailed")
            st.dataframe(st.session_state.df_detailed_display, hide_index=True)


        with st.expander("Show Evaluation Logs", expanded=True):
            # Create a DataFrame to display summary logs
            summary_log_df = pd.DataFrame(evaluation_logs)
            
            # Step 1: Add a new column for checkboxes, initialize with False
            if not summary_log_df.empty:
                if 'selected' not in summary_log_df.columns:
                    summary_log_df['selected'] = False

            # Step 2: Use st.data_editor to make DataFrame editable
            edited_df = st.data_editor(summary_log_df)
            
            # Step 3: Add a delete button
            if st.button("Delete Selected Entries"):
                # Find rows where 'selected' is True and delete them
                selected_rows = edited_df[edited_df['selected'] == True]
                
                for index, row in selected_rows.iterrows():
                    timestamp_to_delete = row['timestamp']
                    filepath_to_delete = row['DataFrame_Detailed']
                    folder_path_to_delete = os.path.dirname(filepath_to_delete)
                    
                    # Remove entry from DataFrame
                    edited_df = edited_df[edited_df['timestamp'] != timestamp_to_delete]
                    
                    # Delete the corresponding folder
                    if os.path.exists(folder_path_to_delete):
                        shutil.rmtree(folder_path_to_delete)

                st.session_state.df_summary_display = None
                st.session_state.df_detailed_display = None

                evaluation_logs = edited_df.to_dict('records')
                save_evaluation_logs(evaluation_logs)

                st.experimental_rerun()
            

            # Selectbox and button to view detailed evaluation
            if evaluation_logs:
                timestamps = [log['timestamp'] for log in evaluation_logs]
                selected_timestamp = st.selectbox("Select Evaluation Timestamp:", timestamps)

                if st.button("View Selected Detailed Evaluation"):
                    selected_row = next(log for log in evaluation_logs if log['timestamp'] == selected_timestamp)

                    st.session_state.df_detailed_display = pd.read_csv(selected_row['DataFrame_Detailed'])

                    st.session_state.df_summary_display = pd.read_csv(selected_row['DataFrame_Summary'])

                    st.experimental_rerun()
                
                if st.button("Hide Entry"):

                    st.session_state.df_summary_display = None
                    st.session_state.df_detailed_display = None
                    st.experimental_rerun()
            

    with test_generator_tab:
        st.header("Generate Synthetic Test Data")

        test_generation_name = st.text_input("Name of the Test Generation")
        if not test_generation_name.strip():  # Check if the evaluation_name is empty or just whitespace
            test_generation_name = "Unnamed_Test_Generation"
        st.session_state.test_generation_name = test_generation_name

        st.session_state.test_size = st.slider("Set Test Size", min_value=1, max_value=30, value=st.session_state.test_size, step=1)

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file:
            generate_dataset_button = st.button("Generate Dataset", key="generate_dataset")
        
            if generate_dataset_button:

                pdf_extractor = PDFTextExtractor(file_path=uploaded_file)
                document_chunks = pdf_extractor.get_pdf_text()

                # Generate the synthetic test dataset using TestsetGenerator
                testset_generator = TestsetGenerator.from_default()
                test_size = st.session_state.test_size  # This can be adjusted or made configurable via the UI
                testset = testset_generator.generate(document_chunks, test_size=test_size)

                file_path = save_to_directory(testset.to_pandas(), test_generation_name)
                st.write(f"Dataset saved to: {file_path}")

                st.dataframe(testset.to_pandas())

                csv_data = testset.to_pandas().to_csv(index=False)
                st.download_button(
                    label="Download Dataset as CSV",
                    data=csv_data,
                    file_name="generated_dataset.csv",
                    mime="text/csv",
                )
                


    st.write(st.session_state.vector_store)
    st.write(st.session_state.use_retriever_model)
    st.write(st.session_state.evaluation_quantity)
    st.write(st.session_state.batch_size)

if __name__ == "__main__":
    main()