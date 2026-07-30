import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

st.set_page_config(page_title="Funny AI Agent", page_icon="🤖")
st.title("🤖 Funny AI Agent")

# ---------------- Session state (same messages list, just kept across reruns) ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a funny AI agent"),
    ]

# ---------------- Render existing chat history ----------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    # SystemMessage is not rendered, same as it never being printed in the terminal version

# ---------------- Chat input (replaces input("You: ")) ----------------
prompt = st.chat_input("You:")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(AIMessage(content=response.content))
    with st.chat_message("assistant"):
        st.markdown(response.content)