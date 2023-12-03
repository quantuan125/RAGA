import streamlit as st

css = '''
<style>
    /* Your existing code */
    .main > div {
            max-width: 100%;
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: {padding_bottom}rem;
        }
    [data-testid="stSidebar"] {
        paddingtop: -200px;
        width: 200px !important; # Set the width to your desired value
    }
    /* Full-width adjustments for text input and button */
    .stTextInput, .stButton > button {
        width: 100% !important;
    }

    /* Additional adjustment for the input element itself */
    [data-testid="stTextInput"] input {
        width: calc(100% - 1rem) !important; /* Adjust the calc value as needed */
    }

    /* Ensure the button aligns with the input */
    [data-testid="stButton"] {
        display: block;
        width: 100% !important;
    }
</style>
'''

def apply_css():
    st.markdown(css, unsafe_allow_html=True)