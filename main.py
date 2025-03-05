from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.set_page_config(page_title="Local LLM")
llm_model = st.selectbox("Choose your LLM model:", ["llama3", "deepseek-r1"])  # Add more options if needed
st.title(llm_model)


# get response

def get_response(query, chat_history, llm_model):
    template = """"
    Answer the question below.

    Here is the conversation history: {chat_history}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=llm_model)
    chain = prompt | model

    return chain.stream({"chat_history": chat_history, "question": query})


# conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# user input
user_query = st.chat_input("Your message")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history, llm_model))

    st.session_state.chat_history.append(AIMessage(ai_response))


