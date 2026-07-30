from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

messages = [
    SystemMessage(content="You are a funny AI agent"),
    
]

print("---------- Welcome (Type 0 to exit) ----------")
#Nur
while True:

    prompt = input("You: ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)