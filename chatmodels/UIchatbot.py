import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

MODE_SETTINGS = {
    "1": ("sad", "You are a sad AI agent. Respond in a sad, empathetic tone."),
    "2": ("angry", "You are an angry AI agent. Respond in an angry, blunt tone."),
    "3": ("funny", "You are a funny AI agent. Respond in a humorous tone."),
    "4": ("happy", "You are a happy AI agent. Respond in a cheerful, positive tone."),
}

MOOD_ICONS = {
    "sad": "😢",
    "funny": "😂",
    "angry": "😡",
    "happy": "😊",
}


def update_system_message(mode_key: str):
    label, prompt_text = MODE_SETTINGS[mode_key]
    for index, message in enumerate(st.session_state.messages):
        if isinstance(message, SystemMessage):
            st.session_state.messages[index] = SystemMessage(content=prompt_text)
            break
    st.session_state.mode = label


# ---------------- Session state (same messages list, just kept across reruns) ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=MODE_SETTINGS["3"][1])]
    st.session_state.mode = "funny"

current_icon = MOOD_ICONS.get(st.session_state.mode, "🤖")
st.set_page_config(page_title="Mood AI Agent", page_icon=current_icon)
st.title(f"{current_icon} Mood-Based AI Agent")

st.info("Mode: Funny (1 = Sad, 2 = Angry, 3 = Funny, 4 = Happy)")
st.caption(f"Current mode: {st.session_state.mode.capitalize()}")

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
    normalized_prompt = prompt.strip()

    if normalized_prompt in MODE_SETTINGS:
        update_system_message(normalized_prompt)
        st.success(f"Mode changed to {st.session_state.mode.capitalize()}")
    else:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        response = model.invoke(st.session_state.messages)

        st.session_state.messages.append(AIMessage(content=response.content))
        with st.chat_message("assistant"):
            st.markdown(response.content)