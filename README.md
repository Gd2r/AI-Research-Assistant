# AI Research Assistant

![Build Status](https://github.com/Gd2r/AI-Research-Assistant/actions/workflows/test.yml/badge.svg)

A Python-based tool to extract, index, and search through PDF research papers using AI embeddings.

## Features

-   **PDF Extraction**: Automatically extracts text from PDF files.
-   **Smart Chunking**: Splits text into manageable chunks for processing.
-   **Semantic Search**: Uses `SentenceTransformers` and `FAISS` to find the most relevant sections based on your query.
-   **Web Interface**: A user-friendly Streamlit app to interact with your documents.
-   **CI/CD Pipeline**: Automated testing with GitHub Actions.

## How It Works

> **The Analogy**: Imagine a library where instead of reading every book to answer a question, you have "index cards" for every paragraph organized by topic. This system creates those index cards (embeddings) so it can find the right paragraph in milliseconds.

For a deep dive into the technical details (Extraction, Chunking, Embeddings, FAISS), check out [ARCHITECTURE.md](ARCHITECTURE.md).

## Challenges & Trade-offs

Building this system required making several architectural decisions:

1.  **FAISS vs. Vector Databases**: I chose **FAISS** (Facebook AI Similarity Search) for its speed and simplicity in local environments. While a full vector database like ChromaDB or Pinecone offers more features (like metadata filtering), FAISS is lightweight and perfect for this scale.
2.  **Chunk Size Strategy**: I settled on a chunk size of **500 words**. Smaller chunks (e.g., 100 words) might miss context, while larger chunks (e.g., 1000 words) dilute the specific meaning of the embedding. 500 words struck the right balance for research papers.
3.  **Model Selection**: I used `all-MiniLM-L6-v2`. It's a "distilled" model, meaning it's much faster and smaller than full BERT models while retaining most of the accuracy. This ensures the app runs smoothly even on standard laptops without GPUs.

## Project Structure

-   `src/`: Source code.
    -   `app.py`: Main Streamlit application.
    -   `extract_text.py`: PDF text extraction logic.
    -   `chunk_text.py`: Text chunking logic.
    -   `embed_text.py`: Embedding generation.
    -   `store_embeddings.py`: FAISS index creation.
    -   `search.py`: Search functionality.
-   `data/`: Directory for your PDF files.
-   `requirements.txt`: Project dependencies.

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Data**:
    Place your PDF files in the `data/` directory.

3.  **Run the App**:
    ```bash
    streamlit run src/app.py
    ```

## Technologies

-   **Python**
-   **Streamlit**
-   **Sentence Transformers** (Hugging Face)
-   **FAISS** (Facebook AI Similarity Search)
-   **pdfplumber**
