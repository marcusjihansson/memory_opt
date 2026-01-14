"""
Embedding service for semantic search capabilities.
"""

import hashlib
import os
import requests
import numpy as np
from typing import List


class EmbeddingService:
    """
    Generate embeddings for semantic search.
    Uses OpenRouter for production embeddings, falls back to mock if no API key.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.dimension = 1536  # text-embedding-3-small dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenRouter.
        Falls back to mock if no API key.
        """
        if self.api_key:
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]
            except Exception as e:
                print(f"OpenRouter embedding failed: {e}, falling back to mock")

        # Mock implementation: hash-based deterministic embedding
        hash_obj = hashlib.sha256(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        embedding = np.random.randn(self.dimension).tolist()

        # Normalize
        norm = np.linalg.norm(embedding)
        return (np.array(embedding) / norm).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts for efficiency"""
        if self.api_key and len(texts) > 1:
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as e:
                print(
                    f"OpenRouter batch embedding failed: {e}, falling back to individual"
                )

        return [self.embed_text(text) for text in texts]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2))
