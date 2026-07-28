

# model = init_chat_model(
#     "gemini-2.5-flash-lite",
#     model_provider="google_genai"
# )
# #print(model)

# response=model.invoke("What is cricketer ?")
# print(response.content)
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "gemini-3.6-flash",
    model_provider="google_genai"
)

response = model.invoke("What is a cricketer?")
print(response.content)