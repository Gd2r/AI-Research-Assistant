# How the AI Research Assistant Works

This document explains the inner workings of the project. It follows a **RAG (Retrieval-Augmented Generation)** pipeline approach (specifically the retrieval component).

## The Big Picture

Imagine you have a library of books (your PDFs). You want to find the answer to a question. Instead of reading every book cover-to-cover every time you have a question, you:
1.  **Read** all the books once.
2.  **Summarize** each paragraph on index cards.
3.  **Organize** those cards by topic.
4.  When you have a question, you just look for the right **index card**.

---

## Step-by-Step Technical Breakdown

### 1. Extraction (`src/extract_text.py`)
**"Reading the Books"**
*   **Tool**: `pdfplumber`
*   **Function**: Opens PDF files and extracts raw text.
*   **Output**: A long string of text for each document.

### 2. Chunking (`src/chunk_text.py`)
**"Creating Index Cards"**
*   **Logic**: Splits the long text into smaller, manageable pieces (e.g., 500 words).
*   **Why**: It allows the AI to focus on specific details rather than getting lost in the entire document.

### 3. Embeddings (`src/embed_text.py`)
**"The Magic Translation"**
*   **Model**: `all-MiniLM-L6-v2` (Sentence Transformer).
*   **Function**: Converts text chunks into **Vectors** (lists of 384 numbers).
*   **Concept**: These numbers represent the *semantic meaning* of the text. Similar concepts have similar numbers.

### 4. Indexing (`src/store_embeddings.py`)
**"The Filing System"**
*   **Tool**: **FAISS** (Facebook AI Similarity Search).
*   **Function**: Stores the vectors in an optimized structure for ultra-fast retrieval.
*   **Metric**: Uses **L2 Distance** (Euclidean) to measure similarity.

### 5. Search (`src/search.py`)
**"Finding the Answer"**
1.  User asks a question.
2.  Question is converted to a vector.
3.  FAISS finds the "nearest neighbor" vectors (chunks) to the question vector.

### 6. The App (`src/app.py`)
**"The Interface"**
*   **Tool**: **Streamlit**.
*   **Function**: Provides a web interface to run the pipeline and display results.

## Understanding Results
*   **Distance Score**: The L2 distance between your query and the result.
*   **Lower is Better**: A distance of `0.0` is a perfect match. `0.8` is very close. `1.5+` is not very relevant.
