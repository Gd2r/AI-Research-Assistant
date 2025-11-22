import streamlit as st
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import glob

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_text import extract_text_from_paper
from chunk_text import chunk_text
from embed_text import embed_chunks
from store_embeddings import store_embeddings
from search import search_query
from config import DATA_DIR, MODEL_NAME

@st.cache_resource
def initialize_system():
    """
    Loads data, processes PDFs, and builds the search index.
    Cached to avoid reloading on every interaction.
    """
    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # data_dir = os.path.join(base_dir, 'data')
    
    pdf_files = glob.glob(os.path.join(DATA_DIR, '*.pdf'))
    if not pdf_files:
        st.error(f"No PDF files found in {DATA_DIR}!")
        return None, None, None

    all_chunks = []
    
    # Process all PDFs
    with st.spinner('Processing PDFs...'):
        for pdf_path in pdf_files:
            text = extract_text_from_paper(pdf_path)
            if text:
                file_chunks = chunk_text(text)
                all_chunks.extend(file_chunks)
    
    if not all_chunks:
        st.error("No text extracted from PDFs.")
        return None, None, None

    with st.spinner('Loading Model and Creating Index...'):
        # Use the modular functions
        embeddings = embed_chunks(all_chunks)
        index = store_embeddings(embeddings)
        
        # We need the model for the search query encoding later
        # Since embed_chunks loads it internally, we might want to load it here or return it.
        # However, search_query expects 'model' to be passed. 
        # Let's instantiate the model here to pass it back, or refactor search_query.
        # For now, let's just load it here to keep the signature consistent.
        model = SentenceTransformer(MODEL_NAME)
        
    return model, index, all_chunks

def main():
    st.title("AI Research Assistant")
    st.write("Ask questions about your PDF documents.")

    model, index, chunks = initialize_system()

    if model and index and chunks:
        query = st.text_input("Ask a question:")

        if query:
            # Perform search
            distances, indices = search_query(query, model, index)
            
            st.subheader("Top Matching Results:")
            for i, idx in enumerate(indices[0]):
                if idx < len(chunks):
                    st.markdown(f"**Result {i+1}** (Distance: {distances[0][i]:.4f})")
                    st.info(chunks[idx])
                else:
                    st.warning("Index out of bounds.")

if __name__ == "__main__":
    main()
