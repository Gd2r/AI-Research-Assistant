import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from extract_text import extract_text_from_paper
    from chunk_text import chunk_text
    from embed_text import embed_chunks
    from store_embeddings import store_embeddings
except ImportError:
    pass

def search_query(query, model, index, k=3):
    """
    Searches the FAISS index for the top k most similar chunks to the query.
    """
    print(f"Searching for: '{query}'")
    # Encode the query
    query_embedding = model.encode([query])
    
    # Search the index
    # D: Distances (lower is better for L2)
    # I: Indices of the nearest neighbors
    D, I = index.search(np.array(query_embedding).astype('float32'), k=k)
    
    return D, I

def main():
    import glob
    from config import DATA_DIR, MODEL_NAME
    
    # Setup paths
    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # data_dir = os.path.join(base_dir, 'data')
    
    # 1. Setup Data (Re-running pipeline to get chunks in memory)
    print("--- Setting up Search Index ---")
    pdf_files = glob.glob(os.path.join(DATA_DIR, '*.pdf'))
    if not pdf_files:
        print("No PDFs found.")
        return
    
    # Use the first PDF
    test_pdf = pdf_files[0]
    text = extract_text_from_paper(test_pdf)
    chunks = chunk_text(text)
    
    # 2. Load Model
    # model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. Create Index
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    # 4. Perform Search
    print("\n--- Performing Search ---")
    query = "What are the benefits of resistance training?"
    distances, indices = search_query(query, model, index)
    
    print(f"\nTop {len(indices[0])} results:")
    for i, idx in enumerate(indices[0]):
        print(f"\nRank {i+1} (Distance: {distances[0][i]:.4f}):")
        print(f"Index: {idx}")
        if idx < len(chunks):
            print(f"Text: {chunks[idx][:200]}...") # Preview chunk text
        else:
            print("Text: [Index out of bounds]")

if __name__ == "__main__":
    main()
