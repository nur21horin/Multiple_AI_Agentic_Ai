# Gen AI Learning

A small Python project demonstrating LangChain-based AI model integration for chat and embeddings.

## Project Overview

This repository contains sample scripts for:
- interacting with chat models via Google Gemini and HuggingFace
- running local HuggingFace chat pipelines
- generating embeddings using HuggingFace sentence-transformers

## Structure

- `requirements.txt` - Python dependencies used by the project.
- `test.py` - quick environment validation that prints the installed LangChain version.
- `chatmodels/`
  - `chat.py` - sample code for initializing a Google Gemini chat model via `langchain`.
  - `huggingface.py` - example of using `langchain_huggingface` with a remote HuggingFace endpoint.
  - `localmodel.py` - example of running a local HuggingFace `text-generation` pipeline.
- `embeddingmodels/`
  - `embeddings.py` - example of generating embeddings with `HuggingFaceEmbeddings`.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

> Note: The project currently uses several LangChain-related packages and HuggingFace integration packages.

## Usage

### 1. Validate environment

Run the quick version test:

```bash
python test.py
```

### 2. Run Google Gemini chat example

Use `chatmodels/chat.py` to initialize and invoke a Google Gemini chat model.

```bash
python chatmodels/chat.py
```

This script expects optional environment configuration via `.env` to be loaded by `python-dotenv`.

### 3. Run HuggingFace endpoint chat example

Use `chatmodels/huggingface.py` to call a HuggingFace endpoint model.

```bash
python chatmodels/huggingface.py
```

### 4. Run local HuggingFace chat pipeline

Use `chatmodels/localmodel.py` to run a local inference pipeline via `HuggingFacePipeline`.

```bash
python chatmodels/localmodel.py
```

### 5. Generate embeddings

Use `embeddingmodels/embeddings.py` to compute document embeddings with the `all-MiniLM-L6-v2` model.

```bash
python embeddingmodels/embeddings.py
```

## Notes

- The repository includes example scripts but is not packaged as a reusable library.
- `python-dotenv` is used in some examples; create a `.env` file if you need to supply credentials or configuration for external providers.
- The HuggingFace examples may require authentication or API access depending on the chosen model and environment.

## Recommended Improvements

- add a `.env.example` file documenting the required environment variables
- add a centralized launcher or CLI for running examples
- add documentation on provider-specific requirements for Google Gemini and HuggingFace

## License

This repository is provided as-is for learning and experimentation.
