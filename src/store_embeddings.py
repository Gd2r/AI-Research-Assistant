import faiss
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from extract_text import extract_text_from_paper
    from chunk_text import chunk_text
    from embed_text import embed_chunks
except ImportError:
    pass

def store_embeddings(embeddings):
    """
    Stores embeddings in a FAISS index for fast retrieval.
    """
    # Convert embeddings to numpy array for FAISS if not already
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index for dimension: {dimension}")
    
    # Create a FAISS index
    # IndexFlatL2 is a brute-force index using Euclidean distance (L2)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings)
    
    print(f"Stored {index.ntotal} embeddings in the index.")
    return index

def main():
    import glob
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Find a PDF to test
    pdf_files = glob.glob(os.path.join(data_dir, '*.pdf'))
    if not pdf_files:
        print("No PDFs found to test.")
        return
        
    test_pdf = pdf_files[0]
    print(f"Processing: {os.path.basename(test_pdf)}")
    
    # 1. Extract
    text = extract_text_from_paper(test_pdf)
    if not text: return
    
    # 2. Chunk
    chunks = chunk_text(text)
    print(f"Generated {len(chunks)} chunks.")
    
    # 3. Embed
    # Embed all chunks this time
    embeddings = embed_chunks(chunks)
    
    # 4. Store
    index = store_embeddings(embeddings)
    
    # Verification
    print(f"Verification: Index contains {index.ntotal} vectors.")
    
    # Optional: Save the index to disk
    index_path = os.path.join(base_dir, 'faiss_index.bin')
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")

if __name__ == "__main__":
    main()
