import streamlit as st
import base64
import os

def display_pdfs(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></embed>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="BR18 Information", page_icon="📚", layout="wide")
    st.title("BR18 Information 📚")

    # Read the PDF file into bytes
    current_directory = os.getcwd()

    # Concatenate the current directory with the relative path to the PDF file
    file_path = os.path.join(current_directory, "BR18.pdf")
    st.write(file_path)

    # Display the PDF
    display_pdfs(file_path)


if __name__ == "__main__":
    main()