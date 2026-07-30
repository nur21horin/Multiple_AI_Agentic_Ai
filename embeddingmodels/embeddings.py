# from dotenv import load_dotenv
# load_dotenv()
# from langchain_openai import OpenAIEmbeddings

# embeddings=OpenAIEmbeddings(
#     model='text-embedding-3-large',
#     dimensions=64
# )
# texts=[
#     "Heloo this is NUR MOHAMMOD",
#     "HEllo your name is GITHUB",
#     "ANd your dimens is 30 ",
#     "You are the backman "
# ]
# vector=embeddings.embed_query("You are going to learn GEN AI")

# print(vector)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    
)

texts=[
    "Heloo this is NUR MOHAMMOD",
    "HEllo your name is GITHUB",
    "ANd your dimens is 30 ",
    "You are the backman "
]
vector = embeddings.embed_documents(texts)

print(vector[:10])  # Show first 10 values
print(len(vector))  # Embedding dimension