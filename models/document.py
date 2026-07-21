"""Document model for RAG system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Document:
    """Represents a document in the RAG system.
    
    Attributes:
        title: Document title.
        url: Document URL.
        content: Document content.
        source_type: Type of source (e.g., 'government', 'news').
        retrieved_at: Timestamp when document was retrieved.
        score: Similarity score from retrieval.
        metadata: Additional metadata dictionary.
    """
    title: str
    url: str
    content: str
    source_type: str
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary.
        
        Returns:
            Dictionary representation of document.
        """
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "source_type": self.source_type,
            "retrieved_at": self.retrieved_at.isoformat(),
            "score": self.score,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create document from dictionary.
        
        Args:
            data: Dictionary containing document data.
            
        Returns:
            Document instance.
        """
        retrieved_at = data.get("retrieved_at")
        if isinstance(retrieved_at, str):
            retrieved_at = datetime.fromisoformat(retrieved_at)
        
        return cls(
            title=data["title"],
            url=data["url"],
            content=data["content"],
            source_type=data["source_type"],
            retrieved_at=retrieved_at or datetime.utcnow(),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
        )
