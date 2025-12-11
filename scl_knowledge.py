#!/usr/bin/env python3
"""
scl_knowledge.py - Source-Grounded Knowledge System

A proper knowledge base that:
1. Stores ORIGINAL source documents (not LLM summaries)
2. Builds semantic embeddings for retrieval
3. Uses SCL at QUERY TIME to verify answers against sources

This is the correct architecture for hallucination-free AI:
- Sources are ground truth
- LLMs synthesize answers FROM sources
- SCL verifies claims are grounded in sources

Usage:
    # Ingest sources
    python scl_knowledge.py ingest --arxiv cs --limit 100
    python scl_knowledge.py ingest --wikipedia "Quantum mechanics"
    
    # Query with source-grounded verification
    python scl_knowledge.py query "What is quantum entanglement?"
    
    # Interactive chat
    python scl_knowledge.py chat
"""

import os
import re
import json
import time
import sqlite3
import hashlib
import logging
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Generator
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-Knowledge")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SourceDocument:
    """An original source document."""
    id: str
    url: str
    title: str
    content: str
    source_type: str  # arxiv, wikipedia, web, file
    timestamp: float
    metadata: Dict = None


@dataclass 
class SourceChunk:
    """A chunk of a source document for retrieval."""
    id: str
    doc_id: str
    text: str
    chunk_index: int
    embedding: np.ndarray = None


@dataclass
class GroundedClaim:
    """A claim with its supporting sources."""
    claim: str
    sources: List[str]  # Source chunk IDs
    confidence: float
    verified: bool


@dataclass
class GroundedResponse:
    """A response grounded in sources."""
    answer: str
    claims: List[GroundedClaim]
    sources_used: List[Dict]
    verification_score: float


# =============================================================================
# Source Store - Stores Original Documents
# =============================================================================

class SourceStore:
    """
    SQLite-based storage for original source documents.
    
    Unlike the atom store, this stores the ACTUAL source text,
    not LLM-generated summaries.
    """
    
    def __init__(self, db_path: str = "scl_sources.db"):
        self.db_path = db_path
        self._embedder = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table - original sources
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT NOT NULL,
                source_type TEXT,
                timestamp REAL,
                metadata TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Chunks table - for retrieval
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                chunk_index INTEGER,
                embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        """)
        
        # Index for fast retrieval
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc 
            ON chunks(doc_id)
        """)
        
        # Full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts 
            USING fts5(text, content='chunks', content_rowid='rowid')
        """)
        
        conn.commit()
        conn.close()
    
    def _get_embedder(self):
        """Lazy load embedder on CPU to avoid GPU memory conflicts."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True,
                device="cpu"  # Use CPU to avoid GPU memory conflicts with SCL server
            )
        return self._embedder
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for retrieval."""
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sent_length = len(sentence)
            
            if current_length + sent_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap
                overlap_text = ' '.join(current_chunk)
                if len(overlap_text) > overlap:
                    # Find sentence boundary for overlap
                    current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sent_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def add_document(self, doc: SourceDocument) -> int:
        """
        Add a source document and create chunks with embeddings.
        
        Returns number of chunks created.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if already exists
            cursor.execute("SELECT id FROM documents WHERE url = ?", (doc.url,))
            if cursor.fetchone():
                logger.info(f"Document already exists: {doc.url}")
                conn.close()
                return 0
            
            # Insert document
            cursor.execute("""
                INSERT INTO documents (id, url, title, content, source_type, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.url, doc.title, doc.content,
                doc.source_type, doc.timestamp,
                json.dumps(doc.metadata) if doc.metadata else None
            ))
            
            # Create chunks
            chunks = self._chunk_text(doc.content)
            
            if not chunks:
                conn.commit()
                conn.close()
                return 0
            
            # Embed chunks
            embedder = self._get_embedder()
            texts_to_embed = [f"search_document: {chunk}" for chunk in chunks]
            embeddings = embedder.encode(texts_to_embed, normalize_embeddings=True)
            
            # Store chunks
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = hashlib.md5(f"{doc.id}:{i}".encode()).hexdigest()[:16]
                
                cursor.execute("""
                    INSERT INTO chunks (id, doc_id, text, chunk_index, embedding)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk_id, doc.id, chunk_text, i,
                    embedding.tobytes()
                ))
            
            conn.commit()
            logger.info(f"Added document '{doc.title}' with {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[SourceChunk, float, Dict]]:
        """
        Semantic search for relevant source chunks.
        
        Returns list of (chunk, similarity_score, document_info).
        """
        embedder = self._get_embedder()
        query_embedding = embedder.encode(
            [f"search_query: {query}"],
            normalize_embeddings=True
        )[0]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all chunks with embeddings
        cursor.execute("""
            SELECT c.id, c.doc_id, c.text, c.chunk_index, c.embedding,
                   d.url, d.title, d.source_type
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE c.embedding IS NOT NULL
        """)
        
        results = []
        for row in cursor.fetchall():
            chunk_id, doc_id, text, chunk_idx, emb_bytes, url, title, source_type = row
            
            # Compute similarity
            chunk_embedding = np.frombuffer(emb_bytes, dtype=np.float32)
            similarity = float(np.dot(query_embedding, chunk_embedding))
            
            if similarity >= threshold:
                chunk = SourceChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    chunk_index=chunk_idx
                )
                doc_info = {
                    "url": url,
                    "title": title,
                    "source_type": source_type
                }
                results.append((chunk, similarity, doc_info))
        
        conn.close()
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """Get store statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT source_type, COUNT(*) FROM documents GROUP BY source_type")
        by_type = dict(cursor.fetchall())
        
        conn.close()
        
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "by_source_type": by_type,
            "database_size_mb": db_size / (1024 * 1024)
        }


# =============================================================================
# Source-Grounded Query Engine
# =============================================================================

class GroundedQueryEngine:
    """
    Query engine that:
    1. Retrieves relevant source chunks
    2. Generates answer grounded in sources
    3. Verifies claims against sources using SCL
    """
    
    def __init__(
        self,
        source_store: SourceStore,
        scl_api_url: str = "http://localhost:8000",
        ollama_url: str = "http://localhost:11434"
    ):
        self.store = source_store
        self.scl_api_url = scl_api_url
        self.ollama_url = ollama_url
    
    def _generate_with_sources(
        self,
        query: str,
        sources: List[Tuple[SourceChunk, float, Dict]],
        model: str = "llama3.1:8b"
    ) -> str:
        """Generate an answer grounded in the provided sources."""
        
        # Build context from sources
        context_parts = []
        for i, (chunk, score, doc_info) in enumerate(sources, 1):
            context_parts.append(
                f"[Source {i}] ({doc_info['title']})\n{chunk.text}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based ONLY on the following sources, answer the question. 
If the sources don't contain enough information, say so.
Always cite which source(s) support each claim you make.

SOURCES:
{context}

QUESTION: {query}

ANSWER (cite sources with [Source N]):"""

        # Generate with Ollama
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temp for factual responses
                    "num_predict": 512
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        return ""
    
    def _verify_against_sources(
        self,
        answer: str,
        sources: List[Tuple[SourceChunk, float, Dict]]
    ) -> Tuple[float, List[Dict]]:
        """
        Verify that claims in the answer are grounded in sources.
        
        Uses SCL ensemble to check each claim.
        """
        # Extract claims from answer
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        source_texts = [chunk.text for chunk, _, _ in sources]
        source_context = "\n".join(source_texts)
        
        verified_claims = []
        total_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Ask SCL to verify this claim against sources
            verification_query = f"""Verify if this claim is supported by the sources.
            
CLAIM: {sentence}

SOURCES:
{source_context}

Is this claim directly supported by the sources? Answer only YES or NO, then explain briefly."""

            try:
                response = requests.post(
                    f"{self.scl_api_url}/v1/truth",
                    json={
                        "query": verification_query,
                        "threshold": 0.7,
                        "use_cache": True
                    },
                    timeout=180
                )
                
                if response.status_code == 200:
                    result = response.json()
                    scl_response = result.get("response", "").lower()
                    surviving = result.get("surviving_atoms", 0)
                    total = result.get("total_atoms", 1)
                    
                    # Check if verified
                    is_verified = "yes" in scl_response[:50] and surviving > 0
                    claim_score = surviving / total if total > 0 else 0
                    
                    verified_claims.append({
                        "claim": sentence,
                        "verified": is_verified,
                        "score": claim_score,
                        "consensus": f"{surviving}/{total}"
                    })
                    
                    if is_verified:
                        total_score += 1
                        
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
                verified_claims.append({
                    "claim": sentence,
                    "verified": False,
                    "score": 0,
                    "error": str(e)
                })
        
        # Calculate overall verification score
        verification_score = total_score / len(verified_claims) if verified_claims else 0
        
        return verification_score, verified_claims
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        verify: bool = True
    ) -> GroundedResponse:
        """
        Answer a question with source-grounded verification.
        
        1. Retrieve relevant sources
        2. Generate answer from sources
        3. Verify claims against sources (optional)
        """
        logger.info(f"Query: {question}")
        
        # Step 1: Retrieve relevant sources
        sources = self.store.search(question, top_k=top_k)
        
        if not sources:
            return GroundedResponse(
                answer="I don't have any relevant sources to answer this question.",
                claims=[],
                sources_used=[],
                verification_score=0
            )
        
        logger.info(f"Found {len(sources)} relevant sources")
        
        # Step 2: Generate answer grounded in sources
        answer = self._generate_with_sources(question, sources)
        
        if not answer:
            return GroundedResponse(
                answer="Failed to generate an answer.",
                claims=[],
                sources_used=[{"url": d["url"], "title": d["title"]} for _, _, d in sources],
                verification_score=0
            )
        
        # Step 3: Verify claims (optional but recommended)
        verification_score = 1.0
        verified_claims = []
        
        if verify:
            logger.info("Verifying claims against sources...")
            verification_score, verified_claims = self._verify_against_sources(
                answer, sources
            )
            logger.info(f"Verification score: {verification_score:.2%}")
        
        return GroundedResponse(
            answer=answer,
            claims=[GroundedClaim(
                claim=c["claim"],
                sources=[],
                confidence=c["score"],
                verified=c["verified"]
            ) for c in verified_claims],
            sources_used=[{
                "url": d["url"],
                "title": d["title"],
                "relevance": score,
                "excerpt": chunk.text[:200] + "..."
            } for chunk, score, d in sources],
            verification_score=verification_score
        )


# =============================================================================
# Source Ingestion
# =============================================================================

class SourceIngester:
    """Ingests sources from various places into the store."""
    
    def __init__(self, store: SourceStore):
        self.store = store
    
    def ingest_arxiv(
        self,
        category: str = "cs",
        limit: int = 100
    ) -> int:
        """Ingest ArXiv paper abstracts."""
        from scl_crawler import ArxivBulkParser
        
        parser = ArxivBulkParser()
        count = 0
        
        logger.info(f"Ingesting ArXiv papers (category: {category}, limit: {limit})")
        
        for doc in parser.harvest_metadata(set_spec=category, limit=limit):
            source_doc = SourceDocument(
                id=doc.id,
                url=doc.url,
                title=doc.title,
                content=doc.content,
                source_type="arxiv",
                timestamp=doc.timestamp,
                metadata=doc.metadata
            )
            
            chunks = self.store.add_document(source_doc)
            if chunks > 0:
                count += 1
                
            if count % 10 == 0:
                logger.info(f"Ingested {count} documents")
        
        return count
    
    def ingest_wikipedia(self, topic: str) -> int:
        """Ingest a Wikipedia article."""
        # Use Wikipedia's standard API with proper headers
        headers = {
            "User-Agent": "SCL-Knowledge/1.0 (https://github.com/scl; research project)"
        }
        
        # Use the action API for extracts
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": topic,
            "prop": "extracts",
            "explaintext": True,
            "format": "json"
        }
        
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract page content
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                logger.error(f"No Wikipedia page found for '{topic}'")
                return 0
            
            page = list(pages.values())[0]
            if "missing" in page:
                logger.error(f"Wikipedia page '{topic}' does not exist")
                return 0
            
            content = page.get("extract", "")
            title = page.get("title", topic)
            
            if not content or len(content) < 100:
                logger.error(f"Wikipedia content too short for '{topic}'")
                return 0
            
            doc = SourceDocument(
                id=hashlib.md5(topic.encode()).hexdigest()[:16],
                url=f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                title=title,
                content=content,
                source_type="wikipedia",
                timestamp=time.time(),
                metadata={"topic": topic}
            )
            
            chunks = self.store.add_document(doc)
            return 1 if chunks > 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to ingest Wikipedia article '{topic}': {e}")
            return 0
    
    def ingest_url(self, url: str) -> int:
        """Ingest content from a URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Simple HTML extraction
            from html.parser import HTMLParser
            
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.title = ""
                    self.in_title = False
                    self.skip = False
                
                def handle_starttag(self, tag, attrs):
                    if tag == 'title':
                        self.in_title = True
                    if tag in ('script', 'style', 'nav', 'footer', 'header', 'aside'):
                        self.skip = True
                
                def handle_endtag(self, tag):
                    if tag == 'title':
                        self.in_title = False
                    if tag in ('script', 'style', 'nav', 'footer', 'header', 'aside'):
                        self.skip = False
                
                def handle_data(self, data):
                    if self.in_title:
                        self.title = data
                    elif not self.skip:
                        self.text.append(data)
            
            extractor = TextExtractor()
            extractor.feed(response.text)
            content = ' '.join(extractor.text)
            content = re.sub(r'\s+', ' ', content).strip()
            
            if len(content) < 100:
                logger.warning(f"Content too short from {url}")
                return 0
            
            doc = SourceDocument(
                id=hashlib.md5(url.encode()).hexdigest()[:16],
                url=url,
                title=extractor.title or url,
                content=content,
                source_type="web",
                timestamp=time.time()
            )
            
            chunks = self.store.add_document(doc)
            return 1 if chunks > 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to ingest URL '{url}': {e}")
            return 0
    
    def ingest_file(self, file_path: str) -> int:
        """Ingest content from a local file."""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return 0
        
        try:
            content = path.read_text(encoding='utf-8')
            
            doc = SourceDocument(
                id=hashlib.md5(file_path.encode()).hexdigest()[:16],
                url=f"file://{path.absolute()}",
                title=path.name,
                content=content,
                source_type="file",
                timestamp=path.stat().st_mtime
            )
            
            chunks = self.store.add_document(doc)
            return 1 if chunks > 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to ingest file '{file_path}': {e}")
            return 0


# =============================================================================
# CLI
# =============================================================================

def cmd_ingest(args, store: SourceStore):
    """Handle ingest command."""
    ingester = SourceIngester(store)
    count = 0
    
    if args.arxiv:
        count = ingester.ingest_arxiv(
            category=args.arxiv,
            limit=args.limit or 100
        )
        print(f"Ingested {count} ArXiv papers")
    
    elif args.wikipedia:
        count = ingester.ingest_wikipedia(args.wikipedia)
        print(f"Ingested {count} Wikipedia article(s)")
    
    elif args.url:
        count = ingester.ingest_url(args.url)
        print(f"Ingested {count} URL(s)")
    
    elif args.file:
        count = ingester.ingest_file(args.file)
        print(f"Ingested {count} file(s)")
    
    return count


def cmd_query(args, store: SourceStore):
    """Handle query command."""
    engine = GroundedQueryEngine(store)
    
    result = engine.query(
        args.question,
        top_k=args.sources or 5,
        verify=not args.no_verify
    )
    
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    
    print("\n" + "=" * 70)
    print("SOURCES USED")
    print("=" * 70)
    for i, source in enumerate(result.sources_used, 1):
        print(f"\n[{i}] {source['title']}")
        print(f"    URL: {source['url']}")
        print(f"    Relevance: {source['relevance']:.2%}")
        print(f"    Excerpt: {source['excerpt']}")
    
    if result.claims:
        print("\n" + "=" * 70)
        print(f"VERIFICATION (Score: {result.verification_score:.0%})")
        print("=" * 70)
        for claim in result.claims:
            status = "✓" if claim.verified else "✗"
            print(f"\n{status} {claim.claim[:100]}")
            print(f"  Confidence: {claim.confidence:.2%}")


def cmd_chat(args, store: SourceStore):
    """Interactive chat mode."""
    engine = GroundedQueryEngine(store)
    
    print("\n" + "=" * 70)
    print("SCL KNOWLEDGE CHAT - Source-Grounded Q&A")
    print("=" * 70)
    print("Ask questions and get answers verified against sources.")
    print("Type 'quit' to exit, 'stats' for database stats.\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'quit':
                break
            
            if question.lower() == 'stats':
                stats = store.get_stats()
                print(f"\nDocuments: {stats['documents']}")
                print(f"Chunks: {stats['chunks']}")
                print(f"By type: {stats['by_source_type']}")
                print(f"Size: {stats['database_size_mb']:.2f} MB\n")
                continue
            
            print("\nSearching sources and generating answer...\n")
            
            result = engine.query(question, verify=True)
            
            print(f"Answer: {result.answer}\n")
            
            if result.sources_used:
                print(f"Sources ({len(result.sources_used)}):")
                for s in result.sources_used[:3]:
                    print(f"  - {s['title']} ({s['relevance']:.0%})")
            
            print(f"\nVerification: {result.verification_score:.0%} of claims verified\n")
            
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def cmd_stats(args, store: SourceStore):
    """Show statistics."""
    stats = store.get_stats()
    
    print("\n" + "=" * 50)
    print("SCL KNOWLEDGE BASE STATISTICS")
    print("=" * 50)
    print(f"Documents:     {stats['documents']:,}")
    print(f"Chunks:        {stats['chunks']:,}")
    print(f"Database size: {stats['database_size_mb']:.2f} MB")
    
    if stats['by_source_type']:
        print("\nBy source type:")
        for source_type, count in stats['by_source_type'].items():
            print(f"  {source_type}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="SCL Knowledge System - Source-Grounded Q&A"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest sources")
    ingest_parser.add_argument("--arxiv", type=str, help="ArXiv category to ingest")
    ingest_parser.add_argument("--wikipedia", type=str, help="Wikipedia topic to ingest")
    ingest_parser.add_argument("--url", type=str, help="URL to ingest")
    ingest_parser.add_argument("--file", type=str, help="File to ingest")
    ingest_parser.add_argument("--limit", type=int, help="Limit for ArXiv ingestion")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--sources", type=int, default=5, help="Number of sources to retrieve")
    query_parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    
    # Global options
    parser.add_argument("--db", type=str, default="scl_sources.db", help="Database path")
    
    args = parser.parse_args()
    
    # Initialize store
    store = SourceStore(args.db)
    
    if args.command == "ingest":
        cmd_ingest(args, store)
    elif args.command == "query":
        cmd_query(args, store)
    elif args.command == "chat":
        cmd_chat(args, store)
    elif args.command == "stats":
        cmd_stats(args, store)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
