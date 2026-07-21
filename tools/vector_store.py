"""Vector store for RAG system using ChromaDB."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.document import Document
from tools.embeddings import get_embedding_model

if TYPE_CHECKING:
    from config.settings import Settings


LOGGER = logging.getLogger(__name__)


class VectorStore:
    """Vector store for document storage and retrieval using ChromaDB."""
    
    def __init__(self, settings: "Settings") -> None:
        """Initialize vector store.
        
        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.chroma_path = Path(settings.chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = get_embedding_model(settings)
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self.collection_name = "govresearch_documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        LOGGER.info(f"Vector store initialized at {self.chroma_path}")
    
    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to vector store.
        
        Args:
            documents: List of documents to add.
            
        Returns:
            Number of documents added (excluding duplicates).
        """
        if not documents:
            return 0
        
        added_count = 0
        
        for doc in documents:
            # Check if URL already exists to avoid duplicates
            existing = self.collection.get(
                where={"url": doc.url},
                limit=1,
            )
            
            if existing["ids"]:
                LOGGER.debug(f"Document already exists, skipping: {doc.url}")
                continue
            
            # Chunk the document content
            chunks = self.text_splitter.split_text(doc.content)
            
            if not chunks:
                LOGGER.warning(f"No chunks generated for document: {doc.url}")
                continue
            
            # Generate embeddings for chunks
            embeddings = self.embedding_model.embed_batch(chunks)
            
            # Prepare metadata for each chunk
            chunk_ids = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.url}_{i}"
                chunk_ids.append(chunk_id)
                
                metadata = {
                    "title": doc.title,
                    "url": doc.url,
                    "source": doc.source_type,
                    "domain": self._extract_domain(doc.url),
                    "retrieved_date":datetime.utcnow().isoformat(),
                    "language": "ko",  # Default to Korean
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                chunk_metadatas.append(metadata)
            
            # Add chunks to collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadatas,
            )
            
            added_count += 1
            LOGGER.debug(f"Added {len(chunks)} chunks for document: {doc.url}")
        
        LOGGER.info(f"Added {added_count} documents to vector store")
        return added_count
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.75,
    ) -> list[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query.
            k: Number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of relevant documents with scores.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        
        if not results["ids"][0]:
            LOGGER.info("No results found in vector store")
            return []
        
        # Convert results to Document objects
        documents = []
        
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to similarity score (cosine distance to similarity)
            score = 1 - distance
            
            # Filter by threshold
            if score < score_threshold:
                LOGGER.debug(f"Document score {score:.3f} below threshold {score_threshold}")
                continue
            
            doc = Document(
                title=metadata["title"],
                url=metadata["url"],
                content=document,
                source_type=metadata["source"],
                score=score,
                metadata=metadata,
            )
            documents.append(doc)
        
        LOGGER.info(f"Retrieved {len(documents)} documents (threshold: {score_threshold})")
        return documents
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL.
        
        Args:
            url: Document URL.
            
        Returns:
            Domain string.
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"
    
    def persist(self) -> None:
        """Persist the vector store to disk."""
        # ChromaDB persistent client auto-saves, but we can log
        LOGGER.info(f"Vector store persisted at {self.chroma_path}")
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        LOGGER.info("Vector store cleared")
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store.
        
        Returns:
            Number of unique documents (by URL).
        """
        # Get all unique URLs
        results = self.collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return 0
        
        unique_urls = set(m["url"] for m in results["metadatas"])
        return len(unique_urls)
