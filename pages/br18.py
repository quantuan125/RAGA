import streamlit as st
import base64
import os
from UI.css import apply_css

@st.cache_data
def display_pdfs(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></embed>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="BR18 Information", page_icon="📚", layout="wide")
    apply_css()
    st.title("BR18 Information 📚")

    # Read the PDF file into bytes
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "pdf", "BR18.pdf")

    # Check if the file exists
    if os.path.exists(file_path):
        # Display the BR18 PDF
        display_pdfs(file_path)
    else:
        st.warning(f"BR18 PDF not found at {file_path}. Please make sure the file is in the correct directory.")


if __name__ == "__main__":
    main()