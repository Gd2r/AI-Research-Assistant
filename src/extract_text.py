import pdfplumber
import os
import glob

def extract_text_from_paper(pdf_path):
    print(f"Extracting text from: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def main():
    # Define the data directory relative to this script
    # Assuming script is in src/ and data is in ../data/
    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # data_dir = os.path.join(base_dir, 'data')
    from config import DATA_DIR
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(DATA_DIR, '*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files.")

    # Process each PDF (or just the first one for the demo)
    for pdf_file in pdf_files:
        text = extract_text_from_paper(pdf_file)
        if text:
            print(f"\n--- Preview of {os.path.basename(pdf_file)} ---")
            print(text[:1000])  # Preview the first 1000 characters
            print("\n" + "="*50 + "\n")
        else:
            print(f"No text extracted from {os.path.basename(pdf_file)}")

if __name__ == "__main__":
    main()
