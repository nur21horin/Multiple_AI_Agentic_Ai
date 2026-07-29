
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "gemini-3.6-flash",
    model_provider="google_genai",
    temperature=0.1,
)

response = model.invoke("What is a Bangladesh?")
print(response.text())
#Nur Mohammod