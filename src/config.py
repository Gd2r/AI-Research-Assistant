import os

# Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index.bin')

# Model Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384

# Processing Configuration
CHUNK_SIZE = 500
