import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunk_text import chunk_text

def test_chunk_text_basic():
    text = "word " * 100
    chunks = chunk_text(text, chunk_size=50)
    assert len(chunks) == 2
    assert len(chunks[0].split()) == 50
    assert len(chunks[1].split()) == 50

def test_chunk_text_remainder():
    text = "word " * 55
    chunks = chunk_text(text, chunk_size=50)
    assert len(chunks) == 2
    assert len(chunks[0].split()) == 50
    assert len(chunks[1].split()) == 5

def test_chunk_text_empty():
    text = ""
    chunks = chunk_text(text, chunk_size=50)
    assert len(chunks) == 0
