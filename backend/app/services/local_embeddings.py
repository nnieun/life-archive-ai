"""Local dense embeddings through Ollama; no external fallback."""

import math

import httpx


class OllamaEmbeddings:
    def __init__(self, model: str, base_url: str, keep_alive: str = '15m'):
        self.model = model
        self.url = base_url.rstrip('/').removesuffix('/v1') + '/api/embed'
        self.keep_alive = keep_alive

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=300, trust_env=False) as client:
            response = client.post(self.url, json={'model': self.model, 'input': texts,
                'truncate': False, 'keep_alive': self.keep_alive})
            response.raise_for_status()
            vectors = response.json().get('embeddings')
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError('Embedding response count is invalid')
        dimension = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError('Embedding vector is invalid')
            dimension = dimension or len(vector)
            if len(vector) != dimension:
                raise ValueError('Embedding dimensions are inconsistent')
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
