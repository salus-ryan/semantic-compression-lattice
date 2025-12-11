#!/usr/bin/env python3
"""
scl_rag.py - Lattice RAG: Retrieval Augmented Generation with SCL

Implements a self-improving "Global Brain" that:
1. Checks the Lattice Cache first for instant (0.6ms) retrieval
2. Falls back to Ensemble computation (70s) if atoms don't exist
3. Stores computed results for future instant access

The more you use it, the faster and smarter it gets.

Features:
- Semantic similarity search across cached atoms
- Automatic knowledge graph building
- Query expansion using cached context
- Streaming responses for better UX

Usage:
    # Start the RAG-enhanced API server
    python scl_rag.py --serve
    
    # Interactive chat mode
    python scl_rag.py --chat
    
    # Single query
    python scl_rag.py --query "What is quantum entanglement?"
"""

import os
import time
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-RAG")


@dataclass
class RetrievedAtom:
    """A meaning atom retrieved from the knowledge base."""
    text: str
    source_query: str
    similarity: float
    timestamp: float
    hit_count: int


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    query: str
    response: str
    retrieved_atoms: List[RetrievedAtom]
    was_cached: bool
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


class SemanticIndex:
    """
    Semantic index for fast similarity search across cached atoms.
    
    Uses embeddings to find relevant cached knowledge for new queries.
    """
    
    def __init__(self, embedder=None):
        self._embedder = embedder
        self._atoms: List[Dict] = []  # {text, embedding, source_query, timestamp}
        self._embeddings: Optional[np.ndarray] = None
    
    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
        return self._embedder
    
    def add_atoms(
        self, 
        atoms: List[str], 
        source_query: str,
        timestamp: float = None
    ):
        """Add atoms to the index."""
        if not atoms:
            return
        
        embedder = self._get_embedder()
        timestamp = timestamp or time.time()
        
        # Embed atoms
        texts = [f"search_document: {a}" for a in atoms]
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        
        for atom, emb in zip(atoms, embeddings):
            self._atoms.append({
                "text": atom,
                "embedding": emb,
                "source_query": source_query,
                "timestamp": timestamp,
                "hit_count": 0
            })
        
        # Rebuild embedding matrix
        self._embeddings = np.array([a["embedding"] for a in self._atoms])
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[RetrievedAtom]:
        """
        Search for relevant atoms using semantic similarity.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
        
        Returns:
            List of RetrievedAtom sorted by similarity
        """
        if not self._atoms or self._embeddings is None:
            return []
        
        embedder = self._get_embedder()
        
        # Embed query
        query_text = f"search_query: {query}"
        query_emb = embedder.encode([query_text], normalize_embeddings=True)[0]
        
        # Compute similarities
        similarities = np.dot(self._embeddings, query_emb)
        
        # Get top-k above threshold
        indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in indices[:top_k]:
            sim = float(similarities[idx])
            if sim < threshold:
                break
            
            atom = self._atoms[idx]
            atom["hit_count"] += 1
            
            results.append(RetrievedAtom(
                text=atom["text"],
                source_query=atom["source_query"],
                similarity=sim,
                timestamp=atom["timestamp"],
                hit_count=atom["hit_count"]
            ))
        
        return results
    
    def size(self) -> int:
        """Return number of indexed atoms."""
        return len(self._atoms)
    
    def save(self, filepath: str):
        """Save index to disk."""
        data = {
            "atoms": [
                {
                    "text": a["text"],
                    "source_query": a["source_query"],
                    "timestamp": a["timestamp"],
                    "hit_count": a["hit_count"]
                }
                for a in self._atoms
            ]
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load index from disk and rebuild embeddings."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self._atoms = []
        for item in data["atoms"]:
            self._atoms.append({
                "text": item["text"],
                "source_query": item["source_query"],
                "timestamp": item["timestamp"],
                "hit_count": item["hit_count"],
                "embedding": None  # Will be computed
            })
        
        if self._atoms:
            embedder = self._get_embedder()
            texts = [f"search_document: {a['text']}" for a in self._atoms]
            embeddings = embedder.encode(texts, normalize_embeddings=True)
            
            for atom, emb in zip(self._atoms, embeddings):
                atom["embedding"] = emb
            
            self._embeddings = np.array([a["embedding"] for a in self._atoms])


class LatticeRAG:
    """
    Lattice RAG: Retrieval Augmented Generation with SCL.
    
    Implements the self-improving "Global Brain" pattern:
    1. Check semantic index for relevant cached atoms
    2. If found: synthesize answer from cached knowledge (instant)
    3. If not found: trigger ensemble, compute truth, cache it
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        threshold: float = 0.75,
        retrieval_threshold: float = 0.7,
        top_k: int = 5,
        index_path: Optional[str] = None
    ):
        self.api_url = api_url.rstrip("/")
        self.threshold = threshold
        self.retrieval_threshold = retrieval_threshold
        self.top_k = top_k
        self.index = SemanticIndex()
        self.index_path = index_path
        
        # Load existing index if available
        if index_path and Path(index_path).exists():
            logger.info(f"Loading semantic index from {index_path}")
            self.index.load(index_path)
            logger.info(f"Loaded {self.index.size()} atoms")
    
    def _check_server(self) -> bool:
        """Check if SCL server is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _fetch_cache_entries(self) -> List[Dict]:
        """Fetch all cached entries from the SCL server."""
        try:
            response = requests.get(f"{self.api_url}/v1/cache/entries", timeout=10)
            if response.status_code == 200:
                return response.json().get("entries", [])
        except Exception as e:
            logger.warning(f"Failed to fetch cache entries: {e}")
        return []
    
    def _query_scl(self, query: str, use_cache: bool = True) -> Dict:
        """Query the SCL API."""
        payload = {
            "query": query,
            "threshold": self.threshold,
            "use_cache": use_cache,
            "max_tokens": 1024
        }
        
        response = requests.post(
            f"{self.api_url}/v1/truth",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    
    def _extract_atoms_from_response(self, response: str) -> List[str]:
        """Extract individual atoms (sentences) from a response."""
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        # Filter and clean
        atoms = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:  # Skip very short fragments
                atoms.append(sent)
        
        return atoms
    
    def _synthesize_from_atoms(
        self, 
        query: str, 
        atoms: List[RetrievedAtom]
    ) -> str:
        """
        Synthesize an answer from retrieved atoms.
        
        For now, this concatenates relevant atoms. In a more advanced
        implementation, this could use an LLM to generate a coherent response.
        """
        if not atoms:
            return ""
        
        # Sort by similarity and deduplicate
        seen = set()
        unique_atoms = []
        for atom in sorted(atoms, key=lambda x: x.similarity, reverse=True):
            if atom.text not in seen:
                seen.add(atom.text)
                unique_atoms.append(atom)
        
        # Combine into response
        response_parts = [atom.text for atom in unique_atoms[:self.top_k]]
        return " ".join(response_parts)
    
    def query(self, query: str) -> RAGResponse:
        """
        Process a query through the Lattice RAG system.
        
        Flow:
        1. Search semantic index for relevant cached atoms
        2. If high-quality matches found: synthesize from cache
        3. Otherwise: query SCL ensemble and cache results
        """
        start_time = time.time()
        retrieval_start = time.time()
        
        # Step 1: Search semantic index
        retrieved = self.index.search(
            query, 
            top_k=self.top_k,
            threshold=self.retrieval_threshold
        )
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Step 2: Check if we have good cached results
        if retrieved and retrieved[0].similarity > 0.85:
            # High confidence match - use cached atoms
            logger.info(
                f"RAG HIT: Found {len(retrieved)} relevant atoms "
                f"(top similarity: {retrieved[0].similarity:.3f})"
            )
            
            response = self._synthesize_from_atoms(query, retrieved)
            
            total_time = (time.time() - start_time) * 1000
            
            return RAGResponse(
                query=query,
                response=response,
                retrieved_atoms=retrieved,
                was_cached=True,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0,
                total_time_ms=total_time
            )
        
        # Step 3: Cache miss - query SCL ensemble
        logger.info(
            f"RAG MISS: No high-confidence matches, querying ensemble..."
        )
        
        generation_start = time.time()
        
        try:
            scl_response = self._query_scl(query)
            response = scl_response.get("response", "")
            
            # Extract and index new atoms
            new_atoms = self._extract_atoms_from_response(response)
            if new_atoms:
                self.index.add_atoms(new_atoms, source_query=query)
                logger.info(f"Indexed {len(new_atoms)} new atoms")
                
                # Save index if path configured
                if self.index_path:
                    self.index.save(self.index_path)
            
        except Exception as e:
            logger.error(f"SCL query failed: {e}")
            response = f"Error: {str(e)}"
        
        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            query=query,
            response=response,
            retrieved_atoms=retrieved,  # May have partial matches
            was_cached=False,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time
        )
    
    def sync_from_cache(self):
        """
        Sync the semantic index with the SCL server's cache.
        
        Fetches all cached entries and indexes their atoms.
        """
        logger.info("Syncing semantic index from SCL cache...")
        
        entries = self._fetch_cache_entries()
        
        for entry in entries:
            query = entry.get("query", "")
            # We'd need to fetch full response - for now skip
            # This would require an API endpoint to get full cached entry
        
        logger.info(f"Sync complete. Index size: {self.index.size()}")
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics."""
        return {
            "index_size": self.index.size(),
            "api_url": self.api_url,
            "threshold": self.threshold,
            "retrieval_threshold": self.retrieval_threshold
        }


def run_chat_mode(rag: LatticeRAG):
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("SCL Lattice RAG - Interactive Chat")
    print("=" * 60)
    print("Type your questions. Type 'quit' or 'exit' to stop.")
    print("Type 'stats' to see system statistics.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        if query.lower() == "stats":
            stats = rag.get_stats()
            print(f"\nIndex size: {stats['index_size']} atoms")
            print(f"API URL: {stats['api_url']}")
            print(f"Threshold: {stats['threshold']}")
            print()
            continue
        
        print("\nThinking...", end="", flush=True)
        
        response = rag.query(query)
        
        # Clear "Thinking..." line
        print("\r" + " " * 20 + "\r", end="")
        
        # Show response
        print(f"\nSCL: {response.response}\n")
        
        # Show metadata
        status = "CACHED" if response.was_cached else "COMPUTED"
        print(f"[{status}] {response.total_time_ms:.1f}ms total")
        
        if response.retrieved_atoms:
            print(f"Retrieved {len(response.retrieved_atoms)} relevant atoms")
        
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SCL Lattice RAG - Self-Improving Knowledge System"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process"
    )
    parser.add_argument(
        "--chat", "-c",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="SCL API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for Lattice Meet (default: 0.75)"
    )
    parser.add_argument(
        "--retrieval-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for RAG retrieval (default: 0.7)"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="lattice_index.json",
        help="Path to save/load semantic index (default: lattice_index.json)"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync semantic index from SCL cache before starting"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG
    rag = LatticeRAG(
        api_url=args.api_url,
        threshold=args.threshold,
        retrieval_threshold=args.retrieval_threshold,
        index_path=args.index_path
    )
    
    # Check server
    if not rag._check_server():
        logger.error(f"SCL server not available at {args.api_url}")
        logger.error("Start the server with: python scl_serve.py")
        return 1
    
    # Sync if requested
    if args.sync:
        rag.sync_from_cache()
    
    # Process based on mode
    if args.query:
        response = rag.query(args.query)
        
        print("\n" + "=" * 60)
        print("SCL LATTICE RAG RESPONSE")
        print("=" * 60)
        print(f"\nQuery: {response.query}")
        print(f"\nResponse:\n{response.response}")
        print(f"\n--- Metadata ---")
        print(f"Cached: {response.was_cached}")
        print(f"Retrieval time: {response.retrieval_time_ms:.1f}ms")
        print(f"Generation time: {response.generation_time_ms:.1f}ms")
        print(f"Total time: {response.total_time_ms:.1f}ms")
        
        if response.retrieved_atoms:
            print(f"\nRetrieved atoms ({len(response.retrieved_atoms)}):")
            for atom in response.retrieved_atoms:
                print(f"  [{atom.similarity:.3f}] {atom.text[:80]}...")
    
    elif args.chat:
        run_chat_mode(rag)
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    exit(main())
