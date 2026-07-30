# Gen AI Learning

A hands-on Python project for learning AI integration with LangChain, HuggingFace, and local inference pipelines.

## What’s in this repository

This repo contains practical examples for:

- Chat-based AI using Google Gemini via `langchain`
- Conversational AI with HuggingFace endpoint models
- Local model inference with HuggingFace pipelines
- Embedding generation using HuggingFace sentence transformers
- Structured output parsing with Mistral and Pydantic

## Repository layout

- `requirements.txt` - Python dependencies for the examples.
- `test.py` - simple environment check that prints the installed `langchain` version.
- `chatmodels/`
  - `chat.py` - Google Gemini chat example using `langchain.chat_models.init_chat_model`.
  - `huggingface.py` - HuggingFace endpoint chat example with `ChatHuggingFace`.
  - `localmodel.py` - local model chat example using `HuggingFacePipeline`.
- `embeddingmodels/`
  - `embeddings.py` - embeddings example using `HuggingFaceEmbeddings` and `sentence-transformers/all-MiniLM-L6-v2`.
- `nur/`
  - `core.py` - structured response extraction example using `ChatMistralAI`, prompts, and `PydanticOutputParser`.
  - `Uicore.py` - Streamlit UI for movie data extraction from free-form text.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If your examples require API keys or provider credentials, add them to a `.env` file in the repository root.

## Running examples

1. Validate the Python environment:

```bash
python test.py
```

2. Run the Google Gemini chat example:

```bash
python chatmodels/chat.py
```

3. Run the HuggingFace endpoint chat example:

```bash
python chatmodels/huggingface.py
```

4. Run the local HuggingFace chat pipeline example:

```bash
python chatmodels/localmodel.py
```

5. Run the embeddings example:

```bash
python embeddingmodels/embeddings.py
```

## Notes

- `chatmodels/huggingface.py` and `chatmodels/localmodel.py` demonstrate two different HuggingFace integration patterns: remote endpoint calls and on-device/local pipeline inference.
- `nur/core.py` and `nur/Uicore.py` show how to build schema-guided extraction with `langchain_core` prompts and `PydanticOutputParser`.
- Some examples rely on external model access, so make sure your environment has the required credentials and dependencies.

## Suggestions

- Add a `.env.example` file with required environment variables.
- Add a simple CLI or launcher to run examples from one place.
- Document provider-specific setup for Google Gemini, HuggingFace endpoints, and local model usage.

## License

This repository is provided for learning and experimentation.
#Nur 
