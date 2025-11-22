# Computer Vision Research Assistant

![Build Status](https://github.com/Gd2r/AI-Research-Assistant/actions/workflows/test.yml/badge.svg)

A Python-based tool to extract, index, and search through **Computer Vision** research papers using AI embeddings.

## Features

-   **PDF Extraction**: Automatically extracts text from PDF files.
-   **Smart Chunking**: Splits text into manageable chunks for processing.
-   **Semantic Search**: Uses `SentenceTransformers` and `FAISS` to find the most relevant sections based on your query.
-   **Web Interface**: A user-friendly Streamlit app to interact with your documents.
-   **CI/CD Pipeline**: Automated testing with GitHub Actions.

## How It Works

> **The Analogy**: Imagine a library where instead of reading every book to answer a question, you have "index cards" for every paragraph organized by topic. This system creates those index cards (embeddings) so it can find the right paragraph in milliseconds.

For a deep dive into the technical details (Extraction, Chunking, Embeddings, FAISS), check out [ARCHITECTURE.md](ARCHITECTURE.md).

## Fine-tuning & Evaluation

This repository includes a pipeline to fine-tune the embedding model on your specific domain (e.g., Computer Vision).

### 1. Fine-tuning
To train the model on your custom dataset (`data/qa_pairs.json`):
```bash
python src/finetune.py
```
This will save a new model to `models/fine_tuned_model`.

### 2. Evaluation
To measure the performance (Recall@k) of the base model vs. the fine-tuned model:
```bash
python src/evaluate.py
```

### Results
On our sample Computer Vision QA dataset (5 pairs), both models achieved:
- **Recall@1**: 1.0000
- **Recall@3**: 1.0000

*Note: With a larger, more complex dataset, you would expect to see the fine-tuned model outperform the base model.*

## UI Screenshots
*(Add screenshots of your Streamlit app here)*

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
