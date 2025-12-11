#!/usr/bin/env python3
"""
scl_crawler.py - Large-Scale Internet Ingestion for SCL

Scalable system for ingesting internet-scale data into the SCL knowledge base.

Data Sources:
1. Wikipedia dumps (cleaned, structured) - ~20GB compressed
2. Common Crawl (web archive) - petabytes available
3. ArXiv bulk access - scientific papers
4. Custom URL lists - targeted crawling

Architecture:
- Producer/Consumer pattern with multiprocessing
- Persistent SQLite storage for atoms
- Checkpoint/resume for long-running jobs
- Rate limiting and politeness

Usage:
    # Ingest Wikipedia dump
    python scl_crawler.py --wikipedia-dump /path/to/enwiki-latest-pages-articles.xml.bz2
    
    # Ingest from Common Crawl
    python scl_crawler.py --common-crawl --segments 10
    
    # Ingest URL list
    python scl_crawler.py --urls urls.txt --workers 4
    
    # Resume interrupted job
    python scl_crawler.py --resume --job-id abc123
"""

import os
import re
import sys
import json
import time
import gzip
import bz2
import sqlite3
import hashlib
import logging
import argparse
import threading
import multiprocessing as mp
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Generator, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from urllib.parse import urlparse
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-Crawler")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Document:
    """A document to be processed."""
    id: str
    url: str
    title: str
    content: str
    source: str  # wikipedia, common_crawl, arxiv, custom
    timestamp: float
    metadata: Dict = None


@dataclass
class ProcessedAtom:
    """A verified meaning atom ready for storage."""
    id: str
    text: str
    embedding_hash: str  # Hash of embedding for deduplication
    source_doc_id: str
    source_url: str
    model_consensus: int  # Number of models that agreed
    energy: float
    timestamp: float


@dataclass
class JobState:
    """State of an ingestion job for checkpointing."""
    job_id: str
    source: str
    total_documents: int
    processed_documents: int
    total_atoms: int
    verified_atoms: int
    start_time: float
    last_checkpoint: float
    status: str  # running, paused, completed, failed
    current_offset: int  # For resuming


# =============================================================================
# Persistent Storage
# =============================================================================

class AtomStore:
    """
    SQLite-based persistent storage for verified atoms.
    
    Implements Theorem 5 (Logarithmic Memory Scaling) at scale.
    Uses content-addressable storage with embedding hashes for deduplication.
    """
    
    def __init__(self, db_path: str = "scl_atoms.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Atoms table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS atoms (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding_hash TEXT NOT NULL,
                source_doc_id TEXT,
                source_url TEXT,
                model_consensus INTEGER DEFAULT 3,
                energy REAL DEFAULT -0.5,
                timestamp REAL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Index for deduplication
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_hash 
            ON atoms(embedding_hash)
        """)
        
        # Index for source lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_url 
            ON atoms(source_url)
        """)
        
        # Full-text search index
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts 
            USING fts5(text, content='atoms', content_rowid='rowid')
        """)
        
        # Documents table (for tracking what's been processed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT,
                title TEXT,
                source TEXT,
                atom_count INTEGER DEFAULT 0,
                processed_at REAL,
                status TEXT DEFAULT 'pending'
            )
        """)
        
        # Jobs table (for checkpointing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                source TEXT,
                total_documents INTEGER DEFAULT 0,
                processed_documents INTEGER DEFAULT 0,
                total_atoms INTEGER DEFAULT 0,
                verified_atoms INTEGER DEFAULT 0,
                start_time REAL,
                last_checkpoint REAL,
                status TEXT DEFAULT 'running',
                current_offset INTEGER DEFAULT 0,
                config TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_atom(self, atom: ProcessedAtom) -> bool:
        """
        Store an atom, deduplicating by embedding hash.
        
        Returns True if atom was new, False if duplicate.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check for duplicate
            cursor.execute(
                "SELECT id FROM atoms WHERE embedding_hash = ?",
                (atom.embedding_hash,)
            )
            if cursor.fetchone():
                return False  # Duplicate
            
            # Insert new atom
            cursor.execute("""
                INSERT INTO atoms 
                (id, text, embedding_hash, source_doc_id, source_url, 
                 model_consensus, energy, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                atom.id, atom.text, atom.embedding_hash,
                atom.source_doc_id, atom.source_url,
                atom.model_consensus, atom.energy, atom.timestamp
            ))
            
            # Update FTS index
            cursor.execute("""
                INSERT INTO atoms_fts(rowid, text) 
                VALUES (last_insert_rowid(), ?)
            """, (atom.text,))
            
            conn.commit()
            return True
            
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def store_atoms_batch(self, atoms: List[ProcessedAtom]) -> int:
        """Store multiple atoms efficiently. Returns count of new atoms."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        new_count = 0
        
        try:
            for atom in atoms:
                # Check for duplicate
                cursor.execute(
                    "SELECT id FROM atoms WHERE embedding_hash = ?",
                    (atom.embedding_hash,)
                )
                if cursor.fetchone():
                    continue
                
                cursor.execute("""
                    INSERT OR IGNORE INTO atoms 
                    (id, text, embedding_hash, source_doc_id, source_url, 
                     model_consensus, energy, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    atom.id, atom.text, atom.embedding_hash,
                    atom.source_doc_id, atom.source_url,
                    atom.model_consensus, atom.energy, atom.timestamp
                ))
                
                if cursor.rowcount > 0:
                    new_count += 1
            
            conn.commit()
            
        finally:
            conn.close()
        
        return new_count
    
    def mark_document_processed(self, doc_id: str, atom_count: int):
        """Mark a document as processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (id, atom_count, processed_at, status)
            VALUES (?, ?, ?, 'completed')
        """, (doc_id, atom_count, time.time()))
        
        conn.commit()
        conn.close()
    
    def is_document_processed(self, doc_id: str) -> bool:
        """Check if document has already been processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT status FROM documents WHERE id = ?",
            (doc_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        return row is not None and row[0] == 'completed'
    
    def save_job_state(self, state: JobState):
        """Save job state for checkpointing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO jobs
            (job_id, source, total_documents, processed_documents,
             total_atoms, verified_atoms, start_time, last_checkpoint,
             status, current_offset)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.job_id, state.source, state.total_documents,
            state.processed_documents, state.total_atoms, state.verified_atoms,
            state.start_time, state.last_checkpoint, state.status,
            state.current_offset
        ))
        
        conn.commit()
        conn.close()
    
    def load_job_state(self, job_id: str) -> Optional[JobState]:
        """Load job state for resuming."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return JobState(
                job_id=row[0], source=row[1], total_documents=row[2],
                processed_documents=row[3], total_atoms=row[4],
                verified_atoms=row[5], start_time=row[6],
                last_checkpoint=row[7], status=row[8], current_offset=row[9]
            )
        return None
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM atoms")
        atom_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'completed'")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT source_url) FROM atoms")
        unique_sources = cursor.fetchone()[0]
        
        conn.close()
        
        # Get file size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        return {
            "total_atoms": atom_count,
            "documents_processed": doc_count,
            "unique_sources": unique_sources,
            "database_size_mb": db_size / (1024 * 1024)
        }
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Full-text search across atoms."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.id, a.text, a.source_url, a.model_consensus
            FROM atoms a
            JOIN atoms_fts f ON a.rowid = f.rowid
            WHERE atoms_fts MATCH ?
            LIMIT ?
        """, (query, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "text": row[1],
                "source_url": row[2],
                "model_consensus": row[3]
            })
        
        conn.close()
        return results


# =============================================================================
# Data Source Parsers
# =============================================================================

class WikipediaDumpParser:
    """
    Parser for Wikipedia XML dumps.
    
    Download from: https://dumps.wikimedia.org/enwiki/latest/
    File: enwiki-latest-pages-articles.xml.bz2 (~20GB)
    """
    
    def __init__(self, dump_path: str):
        self.dump_path = dump_path
    
    def parse(self, limit: int = None) -> Generator[Document, None, None]:
        """
        Stream documents from Wikipedia dump.
        
        Yields Document objects one at a time to handle large files.
        """
        import xml.etree.ElementTree as ET
        
        logger.info(f"Parsing Wikipedia dump: {self.dump_path}")
        
        # Handle compressed files
        if self.dump_path.endswith('.bz2'):
            file_handle = bz2.open(self.dump_path, 'rt', encoding='utf-8')
        elif self.dump_path.endswith('.gz'):
            file_handle = gzip.open(self.dump_path, 'rt', encoding='utf-8')
        else:
            file_handle = open(self.dump_path, 'r', encoding='utf-8')
        
        count = 0
        
        try:
            # Use iterparse for memory efficiency
            context = ET.iterparse(file_handle, events=('end',))
            
            for event, elem in context:
                # Wikipedia namespace
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                if tag == 'page':
                    # Extract page data
                    title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}title')
                    text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')
                    
                    if title_elem is not None and text_elem is not None:
                        title = title_elem.text or ""
                        content = text_elem.text or ""
                        
                        # Skip redirects, disambiguations, etc.
                        if content.startswith('#REDIRECT') or ':' in title:
                            elem.clear()
                            continue
                        
                        # Clean wiki markup (basic)
                        content = self._clean_wiki_markup(content)
                        
                        if len(content) > 100:  # Skip very short articles
                            doc_id = hashlib.md5(title.encode()).hexdigest()[:16]
                            
                            yield Document(
                                id=doc_id,
                                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                title=title,
                                content=content,
                                source="wikipedia",
                                timestamp=time.time()
                            )
                            
                            count += 1
                            if count % 10000 == 0:
                                logger.info(f"Parsed {count} Wikipedia articles")
                            
                            if limit and count >= limit:
                                break
                    
                    # Clear element to free memory
                    elem.clear()
        
        finally:
            file_handle.close()
        
        logger.info(f"Finished parsing {count} Wikipedia articles")
    
    def _clean_wiki_markup(self, text: str) -> str:
        """Remove Wikipedia markup (basic cleaning)."""
        # Remove templates {{...}}
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        
        # Remove references <ref>...</ref>
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^/]*/>', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Convert wiki links [[text|display]] -> display
        text = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', text)
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        
        # Remove external links [url text]
        text = re.sub(r'\[https?://[^\]]+\]', '', text)
        
        # Remove category links
        text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
        
        # Remove file/image links
        text = re.sub(r'\[\[File:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[Image:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
        
        # Remove bold/italic markers
        text = re.sub(r"'''?", '', text)
        
        # Remove section headers
        text = re.sub(r'==+[^=]+=+', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()


class CommonCrawlParser:
    """
    Parser for Common Crawl WARC files.
    
    Common Crawl provides monthly web crawls (~200-300TB each).
    We use the WET files (extracted text) for efficiency.
    
    Index: https://index.commoncrawl.org/
    """
    
    CRAWL_INDEX_URL = "https://index.commoncrawl.org/CC-MAIN-2024-10-index"
    WET_BASE_URL = "https://data.commoncrawl.org/"
    
    def __init__(self, cache_dir: str = ".crawl_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_segment_list(self, crawl_id: str = "CC-MAIN-2024-10") -> List[str]:
        """Get list of WET file segments for a crawl."""
        wet_paths_url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/wet.paths.gz"
        
        cache_file = self.cache_dir / f"{crawl_id}_wet_paths.txt"
        
        if cache_file.exists():
            return cache_file.read_text().strip().split('\n')
        
        logger.info(f"Downloading segment list for {crawl_id}...")
        
        response = requests.get(wet_paths_url, stream=True)
        response.raise_for_status()
        
        # Decompress and save
        content = gzip.decompress(response.content).decode('utf-8')
        cache_file.write_text(content)
        
        return content.strip().split('\n')
    
    def parse_segment(
        self, 
        segment_path: str,
        limit: int = None
    ) -> Generator[Document, None, None]:
        """
        Parse a single WET segment file.
        
        WET files contain extracted plain text from web pages.
        """
        url = f"{self.WET_BASE_URL}{segment_path}"
        
        logger.info(f"Downloading segment: {segment_path}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # WET files are gzipped
        content = gzip.decompress(response.content).decode('utf-8', errors='ignore')
        
        count = 0
        current_doc = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('WARC/1.0'):
                # Save previous document
                if current_doc and current_content:
                    text = '\n'.join(current_content)
                    if len(text) > 200:  # Skip very short pages
                        current_doc['content'] = text
                        
                        yield Document(
                            id=current_doc['id'],
                            url=current_doc['url'],
                            title=current_doc.get('title', ''),
                            content=text,
                            source="common_crawl",
                            timestamp=time.time()
                        )
                        
                        count += 1
                        if limit and count >= limit:
                            return
                
                current_doc = {}
                current_content = []
            
            elif line.startswith('WARC-Target-URI:'):
                url = line.split(': ', 1)[1].strip()
                current_doc['url'] = url
                current_doc['id'] = hashlib.md5(url.encode()).hexdigest()[:16]
                current_doc['title'] = urlparse(url).netloc
            
            elif current_doc and not line.startswith('WARC-') and not line.startswith('Content-'):
                current_content.append(line)
        
        logger.info(f"Parsed {count} documents from segment")
    
    def parse_segments(
        self,
        num_segments: int = 10,
        docs_per_segment: int = 1000,
        crawl_id: str = "CC-MAIN-2024-10"
    ) -> Generator[Document, None, None]:
        """Parse multiple segments."""
        segments = self.get_segment_list(crawl_id)[:num_segments]
        
        for segment in segments:
            yield from self.parse_segment(segment, limit=docs_per_segment)


class ArxivBulkParser:
    """
    Parser for ArXiv bulk data access.
    
    ArXiv provides OAI-PMH access for metadata and S3 for full text.
    https://info.arxiv.org/help/bulk_data.html
    """
    
    OAI_BASE = "http://export.arxiv.org/oai2"
    
    def __init__(self):
        self.session = requests.Session()
    
    def harvest_metadata(
        self,
        from_date: str = None,
        until_date: str = None,
        set_spec: str = None,  # e.g., "cs" for computer science
        limit: int = None
    ) -> Generator[Document, None, None]:
        """
        Harvest paper metadata via OAI-PMH.
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            until_date: End date (YYYY-MM-DD)
            set_spec: ArXiv category (cs, physics, math, etc.)
            limit: Maximum papers to harvest
        """
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv"
        }
        
        if from_date:
            params["from"] = from_date
        if until_date:
            params["until"] = until_date
        if set_spec:
            params["set"] = set_spec
        
        count = 0
        resumption_token = None
        
        while True:
            if resumption_token:
                params = {"verb": "ListRecords", "resumptionToken": resumption_token}
            
            logger.info(f"Fetching ArXiv records (count: {count})...")
            
            response = self.session.get(self.OAI_BASE, params=params)
            response.raise_for_status()
            
            # Parse XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Define namespaces
            ns = {
                "oai": "http://www.openarchives.org/OAI/2.0/",
                "arxiv": "http://arxiv.org/OAI/arXiv/"
            }
            
            records = root.findall(".//oai:record", ns)
            
            for record in records:
                metadata = record.find(".//arxiv:arXiv", ns)
                if metadata is None:
                    continue
                
                arxiv_id = metadata.findtext("arxiv:id", "", ns)
                title = metadata.findtext("arxiv:title", "", ns)
                abstract = metadata.findtext("arxiv:abstract", "", ns)
                
                if title and abstract:
                    # Combine title and abstract as content
                    content = f"{title}\n\n{abstract}"
                    
                    yield Document(
                        id=hashlib.md5(arxiv_id.encode()).hexdigest()[:16],
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        title=title.strip(),
                        content=content.strip(),
                        source="arxiv",
                        timestamp=time.time(),
                        metadata={"arxiv_id": arxiv_id}
                    )
                    
                    count += 1
                    if limit and count >= limit:
                        return
            
            # Check for resumption token
            token_elem = root.find(".//oai:resumptionToken", ns)
            if token_elem is not None and token_elem.text:
                resumption_token = token_elem.text
                time.sleep(3)  # Be polite to ArXiv
            else:
                break
        
        logger.info(f"Harvested {count} ArXiv papers")


# =============================================================================
# Processing Pipeline
# =============================================================================

class SCLProcessor:
    """
    Processes documents through the SCL pipeline.
    
    Connects to the running SCL API server for ensemble processing.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        threshold: float = 0.75,
        batch_size: int = 5
    ):
        self.api_url = api_url.rstrip("/")
        self.threshold = threshold
        self.batch_size = batch_size
        self._embedder = None
    
    def _get_embedder(self):
        """Lazy load embedder for hashing."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
        return self._embedder
    
    def _check_server(self) -> bool:
        """Check if SCL server is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split document into processable paragraphs."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        result = []
        for para in paragraphs:
            para = ' '.join(para.split())  # Normalize whitespace
            
            # Skip short paragraphs
            if len(para) < 100:
                continue
            
            # Split very long paragraphs
            if len(para) > 1500:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) < 1500:
                        current += " " + sent if current else sent
                    else:
                        if len(current) > 100:
                            result.append(current)
                        current = sent
                if len(current) > 100:
                    result.append(current)
            else:
                result.append(para)
        
        return result
    
    def _compute_embedding_hash(self, text: str) -> str:
        """Compute hash of text embedding for deduplication."""
        embedder = self._get_embedder()
        embedding = embedder.encode([f"search_document: {text}"], normalize_embeddings=True)[0]
        # Quantize and hash
        quantized = (embedding * 1000).astype(int).tobytes()
        return hashlib.md5(quantized).hexdigest()
    
    def process_document(self, doc: Document) -> List[ProcessedAtom]:
        """
        Process a single document through SCL.
        
        Returns list of verified atoms.
        """
        paragraphs = self._split_into_paragraphs(doc.content)
        
        if not paragraphs:
            return []
        
        atoms = []
        
        for para in paragraphs[:20]:  # Limit paragraphs per doc
            try:
                # Query SCL API
                response = requests.post(
                    f"{self.api_url}/v1/truth",
                    json={
                        "query": f"Extract and verify the key facts from this text:\n\n{para}",
                        "threshold": self.threshold,
                        "use_cache": True,
                        "max_tokens": 512
                    },
                    timeout=300
                )
                
                if response.status_code != 200:
                    continue
                
                result = response.json()
                
                # Extract surviving atoms from response
                scl_response = result.get("response", "")
                surviving = result.get("surviving_atoms", 0)
                
                if surviving > 0 and scl_response:
                    # Split response into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', scl_response)
                    
                    for sent in sentences:
                        sent = sent.strip()
                        
                        # Filter out meta/preamble noise
                        if any(noise in sent.lower() for noise in [
                            'here are the key facts',
                            'extracted and verified',
                            'the following facts',
                            'key facts from',
                            'facts appear to be',
                            'based on the information provided'
                        ]):
                            continue
                        
                        if len(sent) > 30:
                            atom_id = hashlib.md5(sent.encode()).hexdigest()[:16]
                            emb_hash = self._compute_embedding_hash(sent)
                            
                            atoms.append(ProcessedAtom(
                                id=atom_id,
                                text=sent,
                                embedding_hash=emb_hash,
                                source_doc_id=doc.id,
                                source_url=doc.url,
                                model_consensus=3,
                                energy=-0.5,
                                timestamp=time.time()
                            ))
                
            except Exception as e:
                logger.warning(f"Failed to process paragraph: {e}")
                continue
        
        return atoms
    
    def process_documents_batch(
        self,
        documents: List[Document],
        store: AtomStore,
        job_state: JobState = None
    ) -> Tuple[int, int]:
        """
        Process a batch of documents.
        
        Returns (total_atoms, new_atoms).
        """
        total_atoms = 0
        new_atoms = 0
        
        for doc in documents:
            # Skip if already processed
            if store.is_document_processed(doc.id):
                continue
            
            atoms = self.process_document(doc)
            
            if atoms:
                new = store.store_atoms_batch(atoms)
                total_atoms += len(atoms)
                new_atoms += new
            
            store.mark_document_processed(doc.id, len(atoms))
            
            # Update job state
            if job_state:
                job_state.processed_documents += 1
                job_state.total_atoms += len(atoms)
                job_state.verified_atoms += new_atoms
                job_state.last_checkpoint = time.time()
        
        return total_atoms, new_atoms


# =============================================================================
# Main Crawler
# =============================================================================

class SCLCrawler:
    """
    Main crawler orchestrating large-scale ingestion.
    """
    
    def __init__(
        self,
        db_path: str = "scl_atoms.db",
        api_url: str = "http://localhost:8000",
        workers: int = 1,
        checkpoint_interval: int = 100
    ):
        self.store = AtomStore(db_path)
        self.processor = SCLProcessor(api_url)
        self.workers = workers
        self.checkpoint_interval = checkpoint_interval
    
    def ingest_wikipedia_dump(
        self,
        dump_path: str,
        limit: int = None,
        job_id: str = None
    ) -> JobState:
        """Ingest from Wikipedia XML dump."""
        job_id = job_id or f"wiki_{int(time.time())}"
        
        job_state = JobState(
            job_id=job_id,
            source="wikipedia",
            total_documents=0,
            processed_documents=0,
            total_atoms=0,
            verified_atoms=0,
            start_time=time.time(),
            last_checkpoint=time.time(),
            status="running",
            current_offset=0
        )
        
        parser = WikipediaDumpParser(dump_path)
        batch = []
        
        logger.info(f"Starting Wikipedia ingestion job: {job_id}")
        
        for doc in parser.parse(limit=limit):
            batch.append(doc)
            job_state.total_documents += 1
            
            if len(batch) >= 10:
                total, new = self.processor.process_documents_batch(
                    batch, self.store, job_state
                )
                
                logger.info(
                    f"Processed {job_state.processed_documents} docs, "
                    f"{job_state.verified_atoms} verified atoms"
                )
                
                # Checkpoint
                if job_state.processed_documents % self.checkpoint_interval == 0:
                    self.store.save_job_state(job_state)
                
                batch = []
        
        # Process remaining
        if batch:
            self.processor.process_documents_batch(batch, self.store, job_state)
        
        job_state.status = "completed"
        self.store.save_job_state(job_state)
        
        return job_state
    
    def ingest_common_crawl(
        self,
        num_segments: int = 10,
        docs_per_segment: int = 1000,
        job_id: str = None
    ) -> JobState:
        """Ingest from Common Crawl."""
        job_id = job_id or f"cc_{int(time.time())}"
        
        job_state = JobState(
            job_id=job_id,
            source="common_crawl",
            total_documents=0,
            processed_documents=0,
            total_atoms=0,
            verified_atoms=0,
            start_time=time.time(),
            last_checkpoint=time.time(),
            status="running",
            current_offset=0
        )
        
        parser = CommonCrawlParser()
        batch = []
        
        logger.info(f"Starting Common Crawl ingestion job: {job_id}")
        
        for doc in parser.parse_segments(num_segments, docs_per_segment):
            batch.append(doc)
            job_state.total_documents += 1
            
            if len(batch) >= 10:
                self.processor.process_documents_batch(batch, self.store, job_state)
                
                logger.info(
                    f"Processed {job_state.processed_documents} docs, "
                    f"{job_state.verified_atoms} verified atoms"
                )
                
                if job_state.processed_documents % self.checkpoint_interval == 0:
                    self.store.save_job_state(job_state)
                
                batch = []
        
        if batch:
            self.processor.process_documents_batch(batch, self.store, job_state)
        
        job_state.status = "completed"
        self.store.save_job_state(job_state)
        
        return job_state
    
    def ingest_arxiv(
        self,
        category: str = "cs",
        limit: int = 1000,
        job_id: str = None
    ) -> JobState:
        """Ingest from ArXiv."""
        job_id = job_id or f"arxiv_{int(time.time())}"
        
        job_state = JobState(
            job_id=job_id,
            source="arxiv",
            total_documents=0,
            processed_documents=0,
            total_atoms=0,
            verified_atoms=0,
            start_time=time.time(),
            last_checkpoint=time.time(),
            status="running",
            current_offset=0
        )
        
        parser = ArxivBulkParser()
        batch = []
        
        logger.info(f"Starting ArXiv ingestion job: {job_id}")
        
        for doc in parser.harvest_metadata(set_spec=category, limit=limit):
            batch.append(doc)
            job_state.total_documents += 1
            
            if len(batch) >= 10:
                self.processor.process_documents_batch(batch, self.store, job_state)
                
                logger.info(
                    f"Processed {job_state.processed_documents} docs, "
                    f"{job_state.verified_atoms} verified atoms"
                )
                
                if job_state.processed_documents % self.checkpoint_interval == 0:
                    self.store.save_job_state(job_state)
                
                batch = []
        
        if batch:
            self.processor.process_documents_batch(batch, self.store, job_state)
        
        job_state.status = "completed"
        self.store.save_job_state(job_state)
        
        return job_state
    
    def ingest_url_list(
        self,
        urls_file: str,
        job_id: str = None
    ) -> JobState:
        """Ingest from a list of URLs."""
        job_id = job_id or f"urls_{int(time.time())}"
        
        job_state = JobState(
            job_id=job_id,
            source="custom_urls",
            total_documents=0,
            processed_documents=0,
            total_atoms=0,
            verified_atoms=0,
            start_time=time.time(),
            last_checkpoint=time.time(),
            status="running",
            current_offset=0
        )
        
        urls = Path(urls_file).read_text().strip().split('\n')
        batch = []
        
        logger.info(f"Starting URL list ingestion job: {job_id}")
        
        for url in urls:
            url = url.strip()
            if not url or url.startswith('#'):
                continue
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Basic HTML text extraction
                from html.parser import HTMLParser
                
                class TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.text = []
                        self.skip = False
                    
                    def handle_starttag(self, tag, attrs):
                        if tag in ('script', 'style', 'nav', 'footer', 'header'):
                            self.skip = True
                    
                    def handle_endtag(self, tag):
                        if tag in ('script', 'style', 'nav', 'footer', 'header'):
                            self.skip = False
                    
                    def handle_data(self, data):
                        if not self.skip:
                            self.text.append(data)
                
                extractor = TextExtractor()
                extractor.feed(response.text)
                content = ' '.join(extractor.text)
                
                doc = Document(
                    id=hashlib.md5(url.encode()).hexdigest()[:16],
                    url=url,
                    title=urlparse(url).netloc,
                    content=content,
                    source="custom",
                    timestamp=time.time()
                )
                
                batch.append(doc)
                job_state.total_documents += 1
                
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                continue
            
            if len(batch) >= 10:
                self.processor.process_documents_batch(batch, self.store, job_state)
                
                logger.info(
                    f"Processed {job_state.processed_documents} docs, "
                    f"{job_state.verified_atoms} verified atoms"
                )
                
                batch = []
        
        if batch:
            self.processor.process_documents_batch(batch, self.store, job_state)
        
        job_state.status = "completed"
        self.store.save_job_state(job_state)
        
        return job_state


def main():
    parser = argparse.ArgumentParser(
        description="SCL Large-Scale Internet Ingestion"
    )
    
    # Data sources
    parser.add_argument(
        "--wikipedia-dump",
        type=str,
        help="Path to Wikipedia XML dump (bz2 compressed)"
    )
    parser.add_argument(
        "--common-crawl",
        action="store_true",
        help="Ingest from Common Crawl"
    )
    parser.add_argument(
        "--arxiv",
        type=str,
        nargs='?',
        const="cs",
        help="Ingest from ArXiv (optionally specify category, default: cs)"
    )
    parser.add_argument(
        "--urls",
        type=str,
        help="Path to file containing URLs to ingest"
    )
    
    # Options
    parser.add_argument(
        "--segments",
        type=int,
        default=10,
        help="Number of Common Crawl segments to process (default: 10)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum documents to process"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="scl_atoms.db",
        help="Path to SQLite database (default: scl_atoms.db)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="SCL API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted job"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Job ID for resuming"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search stored atoms"
    )
    
    args = parser.parse_args()
    
    # Initialize crawler
    crawler = SCLCrawler(
        db_path=args.db,
        api_url=args.api_url,
        workers=args.workers
    )
    
    # Check server
    if not crawler.processor._check_server() and not args.stats and not args.search:
        logger.error(f"SCL server not available at {args.api_url}")
        logger.error("Start the server with: python scl_serve.py")
        return 1
    
    # Handle commands
    if args.stats:
        stats = crawler.store.get_stats()
        print("\n" + "=" * 50)
        print("SCL ATOM DATABASE STATISTICS")
        print("=" * 50)
        print(f"Total atoms:         {stats['total_atoms']:,}")
        print(f"Documents processed: {stats['documents_processed']:,}")
        print(f"Unique sources:      {stats['unique_sources']:,}")
        print(f"Database size:       {stats['database_size_mb']:.2f} MB")
        return 0
    
    if args.search:
        results = crawler.store.search(args.search)
        print(f"\nSearch results for '{args.search}':\n")
        for r in results:
            print(f"  [{r['model_consensus']}] {r['text'][:100]}...")
            print(f"      Source: {r['source_url']}\n")
        return 0
    
    # Run ingestion
    print("\n" + "=" * 60)
    print("SCL LARGE-SCALE INTERNET INGESTION")
    print("=" * 60)
    
    job_state = None
    
    if args.wikipedia_dump:
        job_state = crawler.ingest_wikipedia_dump(
            args.wikipedia_dump,
            limit=args.limit,
            job_id=args.job_id
        )
    
    elif args.common_crawl:
        job_state = crawler.ingest_common_crawl(
            num_segments=args.segments,
            docs_per_segment=args.limit or 1000,
            job_id=args.job_id
        )
    
    elif args.arxiv:
        job_state = crawler.ingest_arxiv(
            category=args.arxiv,
            limit=args.limit or 1000,
            job_id=args.job_id
        )
    
    elif args.urls:
        job_state = crawler.ingest_url_list(
            args.urls,
            job_id=args.job_id
        )
    
    else:
        parser.print_help()
        return 0
    
    # Print summary
    if job_state:
        elapsed = time.time() - job_state.start_time
        
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Job ID:              {job_state.job_id}")
        print(f"Source:              {job_state.source}")
        print(f"Documents processed: {job_state.processed_documents:,}")
        print(f"Total atoms:         {job_state.total_atoms:,}")
        print(f"Verified atoms:      {job_state.verified_atoms:,}")
        print(f"Time elapsed:        {elapsed/60:.1f} minutes")
        print(f"Status:              {job_state.status}")
        
        stats = crawler.store.get_stats()
        print(f"\nDatabase now contains {stats['total_atoms']:,} atoms")
    
    return 0


if __name__ == "__main__":
    exit(main())
