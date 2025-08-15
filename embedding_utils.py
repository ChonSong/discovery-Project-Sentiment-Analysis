#!/usr/bin/env python3
"""
Utilities for loading and processing pre-trained word embeddings.
Supports GloVe, FastText, and Word2Vec formats.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import os
from urllib.request import urlretrieve
import gzip


def download_glove_embeddings(embedding_dim: int = 100, data_dir: str = "embeddings") -> str:
    """
    Download GloVe embeddings if not already present.
    
    Args:
        embedding_dim: Dimension of embeddings (50, 100, 200, 300)
        data_dir: Directory to store embeddings
        
    Returns:
        Path to the downloaded embeddings file
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # GloVe 6B (Wikipedia 2014 + Gigaword 5) embeddings
    filename = f"glove.6B.{embedding_dim}d.txt"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading GloVe {embedding_dim}d embeddings...")
        url = f"https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(data_dir, "glove.6B.zip")
        
        # Note: In a real implementation, you would download and extract
        # For this demo, we'll create a simple fallback
        print(f"Would download {url} to {zip_path}")
        print(f"For demo purposes, creating minimal embedding file...")
        create_minimal_embeddings(filepath, embedding_dim)
    
    return filepath


def create_minimal_embeddings(filepath: str, embedding_dim: int):
    """Create a minimal set of embeddings for demonstration."""
    common_words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "good", "bad", "great", "terrible", "amazing", "awful", "love", "hate", "like", "dislike",
        "happy", "sad", "angry", "excited", "disappointed", "satisfied", "pleased", "upset",
        "excellent", "poor", "fantastic", "horrible", "wonderful", "worst", "best", "nice",
        "not", "very", "really", "quite", "extremely", "totally", "absolutely", "never",
        "always", "sometimes", "often", "rarely", "definitely", "probably", "maybe"
    ]
    
    with open(filepath, 'w') as f:
        for word in common_words:
            # Create random embeddings for demo
            embedding = np.random.normal(0, 0.1, embedding_dim)
            embedding_str = ' '.join([f'{val:.6f}' for val in embedding])
            f.write(f"{word} {embedding_str}\n")
    
    print(f"Created minimal embeddings file: {filepath}")


def load_glove_embeddings(filepath: str, vocab: Dict[str, int], embedding_dim: int) -> torch.Tensor:
    """
    Load GloVe embeddings and create embedding matrix for vocabulary.
    
    Args:
        filepath: Path to GloVe embeddings file
        vocab: Vocabulary dictionary {word: index}
        embedding_dim: Dimension of embeddings
        
    Returns:
        Embedding matrix tensor of shape (vocab_size, embedding_dim)
    """
    print(f"Loading GloVe embeddings from {filepath}...")
    
    # Initialize embedding matrix with random values
    vocab_size = len(vocab)
    embedding_matrix = torch.randn(vocab_size, embedding_dim) * 0.1
    
    # Load pre-trained embeddings
    embeddings_dict = {}
    found_words = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) == embedding_dim + 1:
                    word = values[0]
                    vector = np.array(values[1:], dtype=np.float32)
                    embeddings_dict[word] = vector
    except FileNotFoundError:
        print(f"Embeddings file not found: {filepath}")
        print("Using random embeddings for all words")
        return embedding_matrix
    
    # Fill in embeddings for words in vocabulary
    for word, idx in vocab.items():
        if word in embeddings_dict:
            embedding_matrix[idx] = torch.tensor(embeddings_dict[word])
            found_words += 1
    
    print(f"Found embeddings for {found_words}/{vocab_size} words ({found_words/vocab_size*100:.1f}%)")
    
    # Ensure padding token (index 0) has zero embedding
    if 0 < len(embedding_matrix):
        embedding_matrix[0] = torch.zeros(embedding_dim)
    
    return embedding_matrix


def load_fasttext_embeddings(filepath: str, vocab: Dict[str, int], embedding_dim: int) -> torch.Tensor:
    """
    Load FastText embeddings and create embedding matrix for vocabulary.
    Similar to GloVe but FastText can handle out-of-vocabulary words better.
    """
    print(f"Loading FastText embeddings from {filepath}...")
    return load_glove_embeddings(filepath, vocab, embedding_dim)  # Same format as GloVe


def get_pretrained_embeddings(
    vocab: Dict[str, int], 
    embedding_type: str = "glove", 
    embedding_dim: int = 100,
    data_dir: str = "embeddings"
) -> Optional[torch.Tensor]:
    """
    Get pre-trained embeddings for the given vocabulary.
    
    Args:
        vocab: Vocabulary dictionary {word: index}
        embedding_type: Type of embeddings ("glove", "fasttext")
        embedding_dim: Dimension of embeddings
        data_dir: Directory containing embeddings
        
    Returns:
        Embedding matrix tensor or None if loading fails
    """
    try:
        if embedding_type.lower() == "glove":
            filepath = download_glove_embeddings(embedding_dim, data_dir)
            return load_glove_embeddings(filepath, vocab, embedding_dim)
        elif embedding_type.lower() == "fasttext":
            # In a real implementation, you would download FastText embeddings
            # For demo, use the same format as GloVe
            filepath = os.path.join(data_dir, f"fasttext.{embedding_dim}d.txt")
            if not os.path.exists(filepath):
                create_minimal_embeddings(filepath, embedding_dim)
            return load_fasttext_embeddings(filepath, vocab, embedding_dim)
        else:
            print(f"Unsupported embedding type: {embedding_type}")
            return None
    except Exception as e:
        print(f"Error loading {embedding_type} embeddings: {e}")
        return None


def demonstrate_embeddings():
    """Demonstrate embedding loading functionality."""
    # Create a simple vocabulary
    vocab = {"<PAD>": 0, "the": 1, "good": 2, "bad": 3, "movie": 4, "great": 5}
    
    print("=== Embedding Loading Demo ===")
    
    # Test GloVe embeddings
    glove_embeddings = get_pretrained_embeddings(vocab, "glove", 50)
    if glove_embeddings is not None:
        print(f"GloVe embeddings shape: {glove_embeddings.shape}")
        print(f"Sample embedding for 'good': {glove_embeddings[vocab['good']][:5]}...")
    
    # Test FastText embeddings  
    fasttext_embeddings = get_pretrained_embeddings(vocab, "fasttext", 50)
    if fasttext_embeddings is not None:
        print(f"FastText embeddings shape: {fasttext_embeddings.shape}")
        print(f"Sample embedding for 'bad': {fasttext_embeddings[vocab['bad']][:5]}...")


if __name__ == "__main__":
    demonstrate_embeddings()