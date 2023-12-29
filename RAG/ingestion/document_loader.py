import streamlit as st
import tempfile
import os
import pypdf
import pandas as pd
import csv
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from io import StringIO
from langchain.document_loaders import DataFrameLoader


class DocumentLoader:
    
    @staticmethod
    def document_loader_langchain(uploaded_file):
        documents = []
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            original_file_path = tmpfile.name

        try:
            file_name = os.path.splitext(uploaded_file.name)[0]
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            file_type = ""

            if file_extension == '.md':
                file_type = "markdown"
                loader = TextLoader(original_file_path)
                documents = loader.load()

            elif file_extension in ['.csv', '.xlsx', '.xls']:
                file_type = "csv"
                header_row = []
                source_column = None
                
                if file_extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(original_file_path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w+') as temp_csv_file:
                        df.to_csv(temp_csv_file.name, index=False)
                        header_row = df.columns.tolist()
                        file_path_to_load = temp_csv_file.name

                else:
                    file_path_to_load = original_file_path
                    with open(file_path_to_load, newline="", encoding="utf-8") as csvfile:
                        reader = csv.reader(csvfile)
                        header_row = next(reader)

                column_index = st.session_state.get('source_column_index')
                if column_index is not None and column_index < len(header_row):
                    source_column = header_row[column_index]

                # Get metadata columns from session state
                metadata_columns = []
                if st.session_state.get('metadata_column_indexes'):
                    with open(file_path_to_load, newline="", encoding="utf-8") as csvfile:
                        for index in st.session_state.metadata_column_indexes:
                            if index < len(header_row):
                                metadata_columns.append(header_row[index])

                # Load content using CSVLoader with the file path and dynamic source_column
                loader = CSVLoader(file_path_to_load, source_column=source_column, metadata_columns=metadata_columns)
                documents = loader.load()

                if file_extension in ['.xlsx', '.xls']:
                    temp_csv_file.close()
                    os.remove(temp_csv_file.name)


            else:
                raise ValueError("Unsupported file format. Please upload a .md, .pdf, .csv, .xlsx, or .xls file.")

            for doc in documents:
                doc.metadata.update({"file_name": file_name, "file_type": file_type})

        finally:
            os.remove(original_file_path)

        return documents
    
    # @staticmethod
    # def document_loader_unstructured(uploaded_file):
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmpfile:
    #         tmpfile.write(uploaded_file.getvalue())
    #         file_path = tmpfile.name

    #     uploaded_document = partition_md(filename=file_path)

    #     os.remove(file_path)

    #     st.write(uploaded_document)
    #     return uploaded_document