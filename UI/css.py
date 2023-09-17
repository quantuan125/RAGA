import streamlit as st

css = '''
<style>
    /* Your existing code */
    .main > div {
            max-width: {max_width}px;
            padding-top: 2rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }
    [data-testid="stSidebar"] {
        padding-top: -200px;
        width: 300px !important; # Set the width to your desired value
    }
</style>
'''

def apply_css():
    st.markdown(css, unsafe_allow_html=True)