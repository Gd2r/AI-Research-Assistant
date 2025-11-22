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

def test_search_threshold(model):
    # 1. Embed & Index
    embeddings = embed_chunks(MOCK_CHUNKS)
    index = store_embeddings(embeddings)
    
    # 2. Search for something relevant
    query = "machine learning"
    distances, indices = search_query(query, model, index, k=1)
    
    # Distance should be low (e.g., < 1.0 for L2 with normalized vectors, though these aren't normalized)
    # For unnormalized L2, it depends. Let's just check it returns a result.
    assert len(indices[0]) > 0
    
    # 3. Search for something irrelevant
    query = "banana smoothie recipe"
    distances, indices = search_query(query, model, index, k=1)
    
    # Distance should be higher than the relevant query
    # This is a heuristic check
    relevant_dist = search_query("machine learning", model, index, k=1)[0][0][0]
    irrelevant_dist = distances[0][0]
    assert irrelevant_dist > relevant_dist

def test_embedding_dimension_consistency(model):
    text = "Test text"
    embedding = model.encode([text])
    assert embedding.shape[1] == 384
