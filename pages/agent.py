import streamlit as st
from agent.miracle import MRKL
from dotenv import load_dotenv
from utility.sessionstate import Init


def update_custom_db():
    st.session_state.custom_db = not st.session_state.get('custom_db', False)

def update_use_retriever_model():
    st.session_state.use_retriever_model = not st.session_state.get('use_retriever_model', False)

def update_br18_exp():
    st.session_state.br18_exp = not st.session_state.get('br18_exp', False)
    st.session_state.agent = MRKL()

def update_web_search():
    st.session_state.web_search = not st.session_state.get('web_search', False)
    st.session_state.agent = MRKL()

def update_custom_llm_model():
    st.session_state.custom_llm_model = not st.session_state.get('custom_llm_model', False)


def main():
    load_dotenv()
    st.set_page_config(page_title="AGENT SETTINGS", page_icon="🦜️", layout="wide")
    
    with st.empty():
        Init.initialize_session_state()
        Init.initialize_agent_state()

    st.title("Your Agent")


    # Create Tabs
    tool_settings_tab, custom_instruction_tab = st.tabs(["Tool Settings", "Custom Instruction"])


    # Tool Settings Tab
    with tool_settings_tab:
        col1, col2 = st.columns([1, 2])
        with col1:

            st.subheader("Agent Settings")

            custom_llm_model = st.checkbox(
                label="Customize LLM Model", 
                value=st.session_state.get('custom_llm_model', False), 
                help="Toggle to customize the LLM model.",
                key="custom_llm_model_key", 
                on_change=update_custom_llm_model,
                )

            st.subheader("Tool Settings")

            custom_db = st.checkbox(
                label="Customize Document Database Tool", 
                value=st.session_state.get('custom_db', False),
                help="Toggle to enable or disable custom settings for the Document Database Tool.",
                key="custom_db_key", 
                on_change=update_custom_db
            )
            
            br18_experiment = st.checkbox(
                label="Experimental Feature: Enable BR18", 
                value=st.session_state.get('br18_exp', False), 
                help="Toggle to enable or disable BR18 knowledge.",
                key="br18_exp_key", 
                on_change=update_br18_exp,
            )

            if br18_experiment:  # If BR18 is enabled
                search_type = st.radio(
                    "Select Search Type:",
                    options=["By Headers", "By Context"],
                    index=0, horizontal=True  # Default to "By Context"
                )
                st.session_state.search_type = search_type

            web_search_toggle = st.checkbox(
                label="Experimental Feature: Web Search", 
                value=st.session_state.get('web_search', False), 
                help="Toggle to enable or disable the web search feature.",
                key="web_search_key", 
                on_change=update_web_search
            )

            if web_search_toggle:
                selected_num_results = st.slider(
                    "Select Number of Search Results:",
                    min_value=1,
                    max_value=5,
                    value=2
                    )
                st.session_state.websearch_results = selected_num_results
                st.success("Web Search is Enabled.")

        with col2:

            if custom_llm_model:
                with st.expander("Select LLM Model", expanded=True):

                    if 'llm_model_temp' not in st.session_state:
                        st.session_state.llm_model_temp = "gpt-3.5-turbo" 
                        
                    current_llm_model = st.session_state.get('llm_model', "gpt-3.5-turbo")
                    options_list = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"] 
                    
                    default_index = 0
                    if current_llm_model in options_list:
                        default_index = options_list.index(current_llm_model)

                    def update_llm_model():
                        st.session_state.llm_model = st.session_state.llm_model_temp

                    # Create the selectbox and attach the update function via on_change
                    selected_model = st.selectbox(
                        "Choose an LLM model:",
                        options=options_list,
                        index=default_index,
                        key="llm_model_key", 
                        on_change=update_llm_model # The key parameter
                    )

                    def update_streaming_state():
                        st.session_state.streaming = not st.session_state.get('streaming', True)
                        
                    st.checkbox(
                        label="Enable Streaming",
                        value=st.session_state.get('streaming', True),
                        key="streaming_key",
                        on_change=update_streaming_state
                    )
                    
                    # Slider for max_token_limit
                    def update_memory_token_limit():
                        st.session_state.memory_token_limit = st.session_state.memory_token_limit_key
                    
                    st.slider(
                        label="Memory Token Limit",
                        min_value=0,
                        max_value=12000,
                        value=st.session_state.get('memory_token_limit', 1300),
                        step=100,
                        key="memory_token_limit_key",
                        on_change=update_memory_token_limit
                    )

                    def update_ouput_token_limit():
                        st.session_state.ouput_token_limit = st.session_state.ouput_token_limit_key
                    
                    st.slider(
                        label="Output Token Limit",
                        min_value=100,
                        max_value=1000,
                        value=st.session_state.get('ouput_token_limit', 500),
                        step=100,
                        key="ouput_token_limit_key",
                        on_change=update_ouput_token_limit
                    )
                    
                    st.info("Always remember to press 'Save' to activate new settings")
                    if st.button("Save", key="LLM Model"):
                        st.session_state.llm_model = selected_model
                        st.session_state.agent = MRKL()  # Reinitialize the MRKL class
                        st.success("LLM model settings saved and agent reinitialized!")

            if custom_db:
                with st.expander("Document Database Tool Settings", expanded=True):
                    use_retriever_model = st.checkbox(
                        "Use Retriever Model", 
                        value=st.session_state.get('use_retriever_model', False), 
                        key="use_retriever_model_key", 
                        on_change=update_use_retriever_model,
                    )

                    if use_retriever_model:

                        prompt_template = """
                        You are a specialized retriever model. Given the context from the documents below, your task is to:
                        1. Extract in details all relevant pieces of information that answers the query.
                        2. Always prioritize numerical values, names, or specific details over vague and general content.
                        3. Always cite the source for each piece of information you provide using the format: "extracted information" (Source: "File Name", Page: "Page Number").
                        4. If there are no relevant information in the context to the query, explicitly state that.
                        """
                        
                        st.text_area("Prompt for Retriever Model", value=prompt_template, height=200, max_chars=None, key=None, help=None, disabled=True)

                    if st.button("Save", key="Document Database"):
                        st.session_state.agent = MRKL()  # Reinitialize the MRKL class
                        st.success("Settings saved and agent reinitialized!")

            if br18_experiment:
                with st.expander("BR18 Experiment Settings", expanded=True):
                    st.info("This feature is currently under development and not yet available.")

            if web_search_toggle:
                with st.expander("Web Search Settings", expanded=True):
                    st.info("This feature is currently under development and not yet available.")
        
    # Custom Instruction Tab
    with custom_instruction_tab:
        st.header("Custom Instruction")

        system_message_content = st.text_area("System Message", st.session_state.system_message_content, height=200, max_chars=1500)

        formatting_message_content = st.text_area("Formatting Instructions", st.session_state.formatting_message_content, height=200, max_chars=1500)
        
        use_reflection = st.checkbox("Use Reflection Message", value=True, help="Reflection Message are placed at the end of the Prompt")
        if use_reflection is True:
            reflection_message_content = st.text_area("Reflection Message", st.session_state.reflection_message_content, height=200, max_chars=1000)
        else:
            reflection_message_content = ""

        if st.button("Save", key="Custom Instruction"):
            st.session_state.system_message_content = system_message_content
            st.session_state.reflection_message_content = reflection_message_content
            st.session_state.formatting_message_content = formatting_message_content
            st.session_state.agent = MRKL()
            st.success("Settings saved and agent reinitialized!")


    #st.write(st.session_state.custom_db)
    #st.write(st.session_state.use_retriever_model)
    

if __name__== '__main__':
    main()
