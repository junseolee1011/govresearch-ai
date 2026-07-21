"""Embedding model for RAG system."""

import logging
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from config.settings import Settings


LOGGER = logging.getLogger(__name__)


class EmbeddingModel:
    """Singleton embedding model for generating text embeddings."""
    
    _instance: "EmbeddingModel | None" = None
    _model: SentenceTransformer | None = None
    
    def __new__(cls) -> "EmbeddingModel":
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        if self._model is None:
            LOGGER.info(f"Initializing embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            LOGGER.info("Embedding model initialized successfully")
    
    def embed(self, text: str) -> list[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        if self._model is None:
            raise RuntimeError("Embedding model not initialized. Call initialize() first.")
        
        embedding = self._model.encode(text, convert_to_numpy=False)
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        if self._model is None:
            raise RuntimeError("Embedding model not initialized. Call initialize() first.")
        
        embeddings = self._model.encode(texts, convert_to_numpy=False)
        return [emb.tolist() for emb in embeddings]


def get_embedding_model(settings: "Settings | None" = None) -> EmbeddingModel:
    """Get or create the singleton embedding model instance.
    
    Args:
        settings: Application settings (optional).
        
    Returns:
        Singleton embedding model instance.
    """
    model = EmbeddingModel()
    
    if model._model is None and settings is not None:
        model.initialize(settings.embedding_model)
    elif model._model is None:
        model.initialize()
    
    return model
