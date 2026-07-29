from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

model=ChatMistralAI(model="mistral-small-2506",temperature=0.9)

prompt=input("You : ")
response=model.invoke(prompt)

print(response.content) 