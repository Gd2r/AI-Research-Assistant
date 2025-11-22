from sentence_transformers import SentenceTransformer
import os
import sys

# Add current directory to path to allow imports if running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from chunk_text import chunk_text
    from extract_text import extract_text_from_paper
except ImportError:
    pass # Handle imports differently if needed, but sys.path should fix it

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Converts text chunks into embeddings using a SentenceTransformer model.
    
    Model used: 'all-MiniLM-L6-v2'
    - Type: Sentence Transformer
    - Dimensions: 384
    - Use case: Semantic search, clustering
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Generating embeddings...")
    embeddings = model.encode(chunks)
    return embeddings

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
    # We'll just embed the first 3 chunks to save time during this test
    test_chunks = chunks[:3]
    embeddings = embed_chunks(test_chunks)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"First embedding vector (first 5 values): {embeddings[0][:5]}")

if __name__ == "__main__":
    main()
