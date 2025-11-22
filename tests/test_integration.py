import pytest
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embed_text import embed_chunks
from src.store_embeddings import store_embeddings
from src.search import search_query
from sentence_transformers import SentenceTransformer

# Mock data
MOCK_CHUNKS = ["This is a test chunk about AI.", "Another chunk about machine learning.", "Pizza is delicious."]

@pytest.fixture(scope="module")
def model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def test_embedding_shape(model):
    embeddings = model.encode(MOCK_CHUNKS)
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 384

def test_full_pipeline(model):
    # 1. Embed
    embeddings = embed_chunks(MOCK_CHUNKS)
    
    # 2. Index
    index = store_embeddings(embeddings)
    assert index.ntotal == 3
    
    # 3. Search
    query = "artificial intelligence"
    distances, indices = search_query(query, model, index, k=1)
    
    # The first result should be the AI chunk (index 0)
    assert indices[0][0] == 0
