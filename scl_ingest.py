#!/usr/bin/env python3
"""
scl_ingest.py - Recursive Truth Loop: Automated Knowledge Ingestion

Implements automated knowledge base building by:
1. Reading documents (Wikipedia, ArXiv, text files)
2. Splitting into paragraphs
3. Processing through SCL Engine to extract verified atoms
4. Storing invariant atoms in the Lattice Cache

This creates a "pre-digested" knowledge graph where answers are
instant (0.6ms) and mathematically verified.

Usage:
    # Ingest a Wikipedia article
    python scl_ingest.py --wikipedia "Albert Einstein"
    
    # Ingest a text file
    python scl_ingest.py --file document.txt
    
    # Ingest from ArXiv (abstract)
    python scl_ingest.py --arxiv 2301.00001
    
    # Batch ingest from a list
    python scl_ingest.py --batch topics.txt
"""

import os
import re
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict
from datetime import datetime

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-Ingest")


@dataclass
class IngestResult:
    """Result of ingesting a single document."""
    source: str
    title: str
    paragraphs_processed: int
    atoms_extracted: int
    atoms_verified: int
    processing_time_seconds: float
    timestamp: str


@dataclass 
class IngestStats:
    """Aggregate statistics for an ingestion session."""
    documents_processed: int = 0
    total_paragraphs: int = 0
    total_atoms_extracted: int = 0
    total_atoms_verified: int = 0
    total_time_seconds: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DocumentFetcher:
    """Fetches documents from various sources."""
    
    @staticmethod
    def fetch_wikipedia(title: str) -> Optional[Dict]:
        """
        Fetch a Wikipedia article by title.
        
        Returns dict with 'title', 'content', 'url'.
        """
        logger.info(f"Fetching Wikipedia article: {title}")
        
        # Use Wikipedia API
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json"
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    logger.warning(f"Wikipedia article not found: {title}")
                    return None
                
                return {
                    "title": page_data.get("title", title),
                    "content": page_data.get("extract", ""),
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "source": "wikipedia"
                }
        
        except Exception as e:
            logger.error(f"Failed to fetch Wikipedia article: {e}")
            return None
    
    @staticmethod
    def fetch_arxiv(arxiv_id: str) -> Optional[Dict]:
        """
        Fetch an ArXiv paper abstract by ID.
        
        Returns dict with 'title', 'content' (abstract), 'url'.
        """
        logger.info(f"Fetching ArXiv paper: {arxiv_id}")
        
        # Clean up ID format
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Define namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            entry = root.find("atom:entry", ns)
            if entry is None:
                logger.warning(f"ArXiv paper not found: {arxiv_id}")
                return None
            
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            
            title_text = title.text.strip() if title is not None else arxiv_id
            abstract_text = summary.text.strip() if summary is not None else ""
            
            # Clean up whitespace
            title_text = " ".join(title_text.split())
            abstract_text = " ".join(abstract_text.split())
            
            return {
                "title": title_text,
                "content": abstract_text,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "source": "arxiv"
            }
        
        except Exception as e:
            logger.error(f"Failed to fetch ArXiv paper: {e}")
            return None
    
    @staticmethod
    def fetch_file(filepath: str) -> Optional[Dict]:
        """
        Read a local text file.
        
        Returns dict with 'title', 'content', 'url' (filepath).
        """
        logger.info(f"Reading file: {filepath}")
        
        path = Path(filepath)
        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            content = path.read_text(encoding="utf-8")
            return {
                "title": path.stem,
                "content": content,
                "url": str(path.absolute()),
                "source": "file"
            }
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None


class ParagraphSplitter:
    """Splits documents into meaningful paragraphs for processing."""
    
    def __init__(
        self, 
        min_length: int = 50,
        max_length: int = 1000,
        overlap: int = 0
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.overlap = overlap
    
    def split(self, text: str) -> List[str]:
        """
        Split text into paragraphs suitable for SCL processing.
        
        Filters out:
        - Very short paragraphs (< min_length)
        - Section headers
        - References/citations sections
        """
        # Split on double newlines (paragraph breaks)
        raw_paragraphs = re.split(r'\n\s*\n', text)
        
        paragraphs = []
        for para in raw_paragraphs:
            # Clean whitespace
            para = " ".join(para.split())
            
            # Skip if too short
            if len(para) < self.min_length:
                continue
            
            # Skip section headers (typically short, all caps or end with ==)
            if para.isupper() or para.endswith("=="):
                continue
            
            # Skip reference sections
            if para.lower().startswith(("references", "see also", "external links", "notes")):
                continue
            
            # Split long paragraphs
            if len(para) > self.max_length:
                # Split on sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < self.max_length:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk and len(current_chunk) >= self.min_length:
                            paragraphs.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk and len(current_chunk) >= self.min_length:
                    paragraphs.append(current_chunk.strip())
            else:
                paragraphs.append(para)
        
        return paragraphs


class SCLIngester:
    """
    Main ingestion engine that processes documents through SCL.
    
    Connects to the running SCL API server to process paragraphs
    and extract verified meaning atoms.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        threshold: float = 0.75,
        batch_delay: float = 1.0
    ):
        self.api_url = api_url.rstrip("/")
        self.threshold = threshold
        self.batch_delay = batch_delay
        self.fetcher = DocumentFetcher()
        self.splitter = ParagraphSplitter()
        self.stats = IngestStats()
    
    def _check_server(self) -> bool:
        """Check if SCL server is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _process_paragraph(self, paragraph: str, context: str = "") -> Dict:
        """
        Process a single paragraph through the SCL API.
        
        Args:
            paragraph: The text to process
            context: Optional context (e.g., document title)
        
        Returns:
            API response dict
        """
        # Create a query that asks the model to extract facts
        query = f"Extract and verify the key facts from this text:\n\n{paragraph}"
        
        if context:
            query = f"Context: {context}\n\n{query}"
        
        payload = {
            "query": query,
            "threshold": self.threshold,
            "use_cache": True,
            "max_tokens": 512
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/truth",
                json=payload,
                timeout=300  # 5 min timeout for ensemble processing
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}
    
    def ingest_document(self, document: Dict) -> IngestResult:
        """
        Ingest a single document into the SCL knowledge base.
        
        Args:
            document: Dict with 'title', 'content', 'url', 'source'
        
        Returns:
            IngestResult with statistics
        """
        start_time = time.time()
        
        title = document.get("title", "Unknown")
        content = document.get("content", "")
        source = document.get("source", "unknown")
        
        logger.info(f"Ingesting document: {title}")
        
        # Split into paragraphs
        paragraphs = self.splitter.split(content)
        logger.info(f"Split into {len(paragraphs)} paragraphs")
        
        total_atoms_extracted = 0
        total_atoms_verified = 0
        
        for i, para in enumerate(paragraphs):
            logger.info(f"Processing paragraph {i+1}/{len(paragraphs)}...")
            
            result = self._process_paragraph(para, context=title)
            
            if "error" in result:
                logger.warning(f"Paragraph {i+1} failed: {result['error']}")
                continue
            
            atoms_extracted = result.get("total_atoms", 0)
            atoms_verified = result.get("surviving_atoms", 0)
            
            total_atoms_extracted += atoms_extracted
            total_atoms_verified += atoms_verified
            
            cached = result.get("cached", False)
            status = "CACHED" if cached else "COMPUTED"
            
            logger.info(
                f"  [{status}] {atoms_verified}/{atoms_extracted} atoms verified"
            )
            
            # Rate limiting
            if not cached and i < len(paragraphs) - 1:
                time.sleep(self.batch_delay)
        
        elapsed = time.time() - start_time
        
        result = IngestResult(
            source=source,
            title=title,
            paragraphs_processed=len(paragraphs),
            atoms_extracted=total_atoms_extracted,
            atoms_verified=total_atoms_verified,
            processing_time_seconds=elapsed,
            timestamp=datetime.now().isoformat()
        )
        
        # Update aggregate stats
        self.stats.documents_processed += 1
        self.stats.total_paragraphs += len(paragraphs)
        self.stats.total_atoms_extracted += total_atoms_extracted
        self.stats.total_atoms_verified += total_atoms_verified
        self.stats.total_time_seconds += elapsed
        
        return result
    
    def ingest_wikipedia(self, title: str) -> Optional[IngestResult]:
        """Ingest a Wikipedia article by title."""
        doc = self.fetcher.fetch_wikipedia(title)
        if doc:
            return self.ingest_document(doc)
        return None
    
    def ingest_arxiv(self, arxiv_id: str) -> Optional[IngestResult]:
        """Ingest an ArXiv paper abstract by ID."""
        doc = self.fetcher.fetch_arxiv(arxiv_id)
        if doc:
            return self.ingest_document(doc)
        return None
    
    def ingest_file(self, filepath: str) -> Optional[IngestResult]:
        """Ingest a local text file."""
        doc = self.fetcher.fetch_file(filepath)
        if doc:
            return self.ingest_document(doc)
        return None
    
    def ingest_batch(self, topics_file: str) -> List[IngestResult]:
        """
        Ingest multiple topics from a file.
        
        File format (one per line):
            wikipedia:Albert Einstein
            arxiv:2301.00001
            file:/path/to/document.txt
        """
        results = []
        
        path = Path(topics_file)
        if not path.exists():
            logger.error(f"Batch file not found: {topics_file}")
            return results
        
        lines = path.read_text().strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            try:
                if line.startswith("wikipedia:"):
                    title = line[len("wikipedia:"):]
                    result = self.ingest_wikipedia(title)
                elif line.startswith("arxiv:"):
                    arxiv_id = line[len("arxiv:"):]
                    result = self.ingest_arxiv(arxiv_id)
                elif line.startswith("file:"):
                    filepath = line[len("file:"):]
                    result = self.ingest_file(filepath)
                else:
                    # Default to Wikipedia
                    result = self.ingest_wikipedia(line)
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to process '{line}': {e}")
                self.stats.errors.append(f"{line}: {str(e)}")
        
        return results
    
    def get_stats(self) -> Dict:
        """Get aggregate ingestion statistics."""
        return asdict(self.stats)


def main():
    parser = argparse.ArgumentParser(
        description="SCL Recursive Truth Loop - Automated Knowledge Ingestion"
    )
    
    parser.add_argument(
        "--wikipedia", "-w",
        type=str,
        help="Wikipedia article title to ingest"
    )
    parser.add_argument(
        "--arxiv", "-a",
        type=str,
        help="ArXiv paper ID to ingest (e.g., 2301.00001)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Local text file to ingest"
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Batch file with list of sources to ingest"
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
        "--delay",
        type=float,
        default=1.0,
        help="Delay between paragraphs in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for ingestion results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Initialize ingester
    ingester = SCLIngester(
        api_url=args.api_url,
        threshold=args.threshold,
        batch_delay=args.delay
    )
    
    # Check server
    if not ingester._check_server():
        logger.error(f"SCL server not available at {args.api_url}")
        logger.error("Start the server with: python scl_serve.py")
        return 1
    
    logger.info("=" * 60)
    logger.info("SCL Recursive Truth Loop - Knowledge Ingestion")
    logger.info("=" * 60)
    
    results = []
    
    # Process based on arguments
    if args.wikipedia:
        result = ingester.ingest_wikipedia(args.wikipedia)
        if result:
            results.append(result)
    
    elif args.arxiv:
        result = ingester.ingest_arxiv(args.arxiv)
        if result:
            results.append(result)
    
    elif args.file:
        result = ingester.ingest_file(args.file)
        if result:
            results.append(result)
    
    elif args.batch:
        results = ingester.ingest_batch(args.batch)
    
    else:
        parser.print_help()
        return 0
    
    # Print summary
    stats = ingester.get_stats()
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Total paragraphs:    {stats['total_paragraphs']}")
    print(f"Atoms extracted:     {stats['total_atoms_extracted']}")
    print(f"Atoms verified:      {stats['total_atoms_verified']}")
    print(f"Total time:          {stats['total_time_seconds']:.1f}s")
    
    if stats['total_atoms_extracted'] > 0:
        verification_rate = stats['total_atoms_verified'] / stats['total_atoms_extracted']
        print(f"Verification rate:   {verification_rate:.1%}")
    
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats['errors']:
            print(f"  - {err}")
    
    # Save results if output specified
    if args.output:
        output_data = {
            "stats": stats,
            "results": [asdict(r) for r in results],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
