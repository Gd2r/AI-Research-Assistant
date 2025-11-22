import os
import glob
# Import the extraction function from the sibling script
try:
    from extract_text import extract_text_from_paper
except ImportError:
    # Fallback if running directly from src/
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extract_text import extract_text_from_paper

def chunk_text(text, chunk_size=500):
    """
    Splits text into chunks of approximately `chunk_size` words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Find a PDF to test with
    pdf_files = glob.glob(os.path.join(data_dir, '*.pdf'))
    
    if not pdf_files:
        print("No PDFs found to test chunking.")
        return

    # Use the first PDF found
    test_pdf = pdf_files[0]
    print(f"Testing chunking on: {os.path.basename(test_pdf)}")
    
    # Extract text
    text = extract_text_from_paper(test_pdf)
    if not text:
        print("No text extracted.")
        return

    # Chunk text
    chunks = chunk_text(text, chunk_size=500)
    print(f"Total chunks created: {len(chunks)}")
    
    if chunks:
        print(f"\n--- First Chunk Preview (approx 500 words) ---")
        print(chunks[0][:1000] + "...") # Print first 1000 chars of the chunk
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
