from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage


# Initialize chat history and model list in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_list" not in st.session_state:
    st.session_state.model_list = ["llama3", "deepseek-r1", "llama3.2-vision"]  # Predefined model list
if "custom_model" not in st.session_state:  # To hold the custom model name
    st.session_state.custom_model = ''

st.set_page_config(page_title="Local LLM")

# Sidebar for model selection and input
st.sidebar.header("Model Management")

# Dropdown for selecting a model
llm_model = st.sidebar.selectbox("Choose your LLM model:", st.session_state.model_list)

# Function to handle submission
def submit():
    if st.session_state.custom_model and st.session_state.custom_model not in st.session_state.model_list:
        st.session_state.model_list.append(st.session_state.custom_model)
        st.success(f"Model '{st.session_state.custom_model}' added.")
        st.session_state.custom_model = ''  # Clear the input box

# Input for custom model name
st.sidebar.text_input('Or enter your custom LLM model name:', key='custom_model', on_change=submit)

# Add custom model button
if st.sidebar.button("Add Model"):
    submit()  # Call the submit function when the button is clicked


# Remove selected model button
if st.sidebar.button("Remove Selected Model"):
    if llm_model in st.session_state.model_list:
        st.session_state.model_list.remove(llm_model)
        st.success(f"Model '{llm_model}' removed.")
        # Reset selected model if it was removed
        if llm_model == llm_model:
            llm_model = st.session_state.model_list[0] if st.session_state.model_list else None
    else:
        st.warning("The selected model is not in the list.")

# Determine the model to use
model_to_use = llm_model
if(model_to_use == None):
    st.title("Welcome")
else:
    st.title(model_to_use)

# Get response
def get_response(query, chat_history, model_to_use):
    template = """
    Answer the question below.

    Here is the conversation history: {chat_history}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=model_to_use)
    chain = prompt | model


    return chain.stream({"chat_history": chat_history, "question": query})

# Conversation display
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("Your message")
if user_query:
    # Append user message to chat history
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    try:
        with st.chat_message("AI"):
            ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history, model_to_use))
        
        # Append AI response to chat history
        st.session_state.chat_history.append(AIMessage(ai_response))

    except Exception as e:
        st.error(f"An error occurred. Please ensure your model name is correct or try a different one.")
