"""Vector store for RAG system using ChromaDB."""

import logging
import re
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


def preprocess_document_content(content: str) -> str:
    """Preprocess document content to remove TOC, page numbers, and OCR noise.
    
    Args:
        content: Raw document content.
        
    Returns:
        Cleaned content.
    """
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Skip page numbers (standalone numbers)
        if re.match(r'^\d{1,4}$', stripped):
            continue
        
        # Skip page number patterns
        if re.match(r'^Page\s*\d+', stripped, re.IGNORECASE):
            continue
        if re.match(r'^페이지\s*\d+', stripped):
            continue
        if re.match(r'^P\.\s*\d+', stripped):
            continue
        
        # Skip TOC markers
        if stripped in ['CONTENTS', '목차', '차례', 'INDEX', '목 차']:
            continue
        
        # Skip lines that are only OCR noise (special characters, numbers)
        if re.match(r'^[\d\s\-\_\.]+$', stripped):
            continue
        
        # Skip image caption markers
        if re.match(r'^<.*?체계도.*?>$', stripped):
            continue
        if re.match(r'^<.*?소개자료.*?>$', stripped):
            continue
        if re.match(r'^<.*?이미지.*?>$', stripped):
            continue
        if re.match(r'^\[.*?그림.*?\]$', stripped):
            continue
        
        # Skip source references
        if re.match(r'^<출처', stripped):
            continue
        if re.match(r'^출처\s*:', stripped):
            continue
        
        cleaned_lines.append(line)
    
    # Join and clean up extra whitespace
    cleaned_content = '\n'.join(cleaned_lines)
    cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)
    
    return cleaned_content.strip()


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
        
        # Initialize text splitter for service-based chunking
        # Changed from character-based to service-based chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # 400-800 tokens target
            chunk_overlap=100,  # 50-100 tokens overlap
            length_function=len,
            separators=["\n\n### ", "\n\n## ", "\n\n", "\n", " "],  # Split by headers first
        )
        
        # Print chunk generation rules
        LOGGER.info("=" * 60)
        LOGGER.info("Chunk Generation Rules:")
        LOGGER.info(f"  Splitter Type: RecursiveCharacterTextSplitter")
        LOGGER.info(f"  Chunk Size: {self.text_splitter._chunk_size} characters")
        LOGGER.info(f"  Chunk Overlap: {self.text_splitter._chunk_overlap} characters")
        LOGGER.info(f"  Separators: {self.text_splitter._separators}")
        LOGGER.info(f"  Recursive: Yes")
        LOGGER.info("=" * 60)
        
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
            
            # Preprocess document content to remove TOC, page numbers, OCR noise
            cleaned_content = preprocess_document_content(doc.content)
            
            if not cleaned_content:
                LOGGER.warning(f"Empty content after preprocessing for document: {doc.url}")
                continue
            
            # Chunk the document content
            chunks = self.text_splitter.split_text(cleaned_content)
            
            if not chunks:
                LOGGER.warning(f"No chunks generated for document: {doc.url}")
                continue
            
            # Save chunks to txt files for debugging
            chunk_output_dir = Path("debug_chunks")
            chunk_output_dir.mkdir(exist_ok=True)
            
            # Analyze chunks for CONTENTS, multiple services, OCR issues
            chunk_stats = {
                'total': len(chunks),
                'with_contents': 0,
                'with_multiple_services': 0,
                'with_ocr_issues': 0,
                'lengths': []
            }
            
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.url}_{i}"
                
                # Save chunk to txt file
                chunk_file = chunk_output_dir / f"chunk_{chunk_id.replace(':', '_').replace('/', '_')}.txt"
                chunk_file.write_text(chunk, encoding='utf-8')
                
                # Analyze chunk
                chunk_stats['lengths'].append(len(chunk))
                
                # Check for CONTENTS
                if 'CONTENTS' in chunk or '목차' in chunk or '차례' in chunk:
                    chunk_stats['with_contents'] += 1
                    LOGGER.warning(f"[Chunk Analysis] Chunk #{i} contains CONTENTS/목차 - REMOVING")
                    continue  # Skip this chunk
                
                # Check for OCR issues (standalone numbers, special characters)
                lines = chunk.split('\n')
                ocr_noise_count = sum(1 for line in lines if re.match(r'^[\d\s\-\_\.]+$', line.strip()))
                if ocr_noise_count > len(lines) * 0.5:
                    chunk_stats['with_ocr_issues'] += 1
                    LOGGER.warning(f"[Chunk Analysis] Chunk #{i} has OCR noise - REMOVING")
                    continue  # Skip this chunk
                
                # Check for multiple services (AI + 시스템/서비스/플랫폼/모델/솔루션 patterns)
                service_patterns = [r'AI.*시스템', r'AI.*서비스', r'AI.*플랫폼', r'AI.*모델', r'AI.*솔루션']
                service_count = sum(len(re.findall(pattern, chunk)) for pattern in service_patterns)
                if service_count >= 2:
                    chunk_stats['with_multiple_services'] += 1
                    LOGGER.warning(f"[Chunk Analysis] Chunk #{i} has {service_count} services - SPLIT REQUIRED")
                    # Still include the chunk but log warning
                
                valid_chunks.append(chunk)
            
            # Replace original chunks with filtered valid chunks
            chunks = valid_chunks
            
            # Print chunk statistics
            if chunk_stats['total'] > 0:
                avg_length = sum(chunk_stats['lengths']) / len(chunk_stats['lengths'])
                max_length = max(chunk_stats['lengths'])
                LOGGER.info("=" * 60)
                LOGGER.info(f"Chunk Statistics for {doc.url}:")
                LOGGER.info(f"  Total Chunks: {chunk_stats['total']}")
                LOGGER.info(f"  Valid Chunks (after filtering): {len(chunks)}")
                LOGGER.info(f"  Removed (CONTENTS): {chunk_stats['with_contents']}")
                LOGGER.info(f"  Removed (OCR Issues): {chunk_stats['with_ocr_issues']}")
                LOGGER.info(f"  With Multiple Services: {chunk_stats['with_multiple_services']}")
                LOGGER.info(f"  Average Length: {avg_length:.0f} characters")
                LOGGER.info(f"  Max Length: {max_length} characters")
                LOGGER.info("=" * 60)
            
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
