#!/usr/bin/env python3
"""
scl_serve.py - SCL Truth API Server

FastAPI wrapper for the Semantic Compression Lattice engine.
Keeps models loaded in VRAM and implements Lattice Caching (Theorem 5).

Endpoints:
    POST /v1/truth - Query the SCL engine
    GET /v1/cache/stats - View cache statistics
    DELETE /v1/cache - Clear the lattice cache
    GET /health - Health check

Usage:
    uvicorn scl_serve:app --host 0.0.0.0 --port 8000
"""

import os
import time
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scl_engine import SCLEngine, SCLConfig, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-Serve")


# =============================================================================
# Lattice Cache (Theorem 5: Logarithmic Memory Scaling)
# =============================================================================

@dataclass
class CachedLattice:
    """
    Cached result of a Lattice Meet computation.
    
    Implements Theorem 5: If the number of distinct meaning-atoms grows o(n)
    along a reasoning trace of length n, then mem(n) ∈ O(log n).
    
    By caching computed lattices, we reuse existing Meaning Atoms.
    """
    query_hash: str
    query: str
    response: str
    surviving_atoms: int
    total_atoms: int
    model_outputs: Dict[str, str]
    filtered: Dict[str, List[str]]
    timestamp: float
    hit_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LatticeCache:
    """
    In-memory cache for computed Lattice Meet results.
    
    Uses query hashing for O(1) lookup. Implements LRU eviction
    when cache exceeds max_size.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CachedLattice] = {}
        self._access_order: List[str] = []
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str, threshold: float) -> str:
        """Create a unique hash for query + parameters."""
        key = f"{query.lower().strip()}|{threshold}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def get(self, query: str, threshold: float) -> Optional[CachedLattice]:
        """Retrieve cached lattice if exists."""
        key = self._hash_query(query, threshold)
        
        if key in self._cache:
            self.hits += 1
            cached = self._cache[key]
            cached.hit_count += 1
            # Move to end of access order (LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return cached
        
        self.misses += 1
        return None
    
    def put(
        self,
        query: str,
        threshold: float,
        response: str,
        metadata: Dict
    ) -> CachedLattice:
        """Store a computed lattice in cache."""
        key = self._hash_query(query, threshold)
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
        
        cached = CachedLattice(
            query_hash=key,
            query=query,
            response=response,
            surviving_atoms=metadata.get("surviving", 0),
            total_atoms=metadata.get("total_atoms", 0),
            model_outputs=metadata.get("model_outputs", {}),
            filtered=metadata.get("filtered", {}),
            timestamp=time.time()
        )
        
        self._cache[key] = cached
        self._access_order.append(key)
        
        return cached
    
    def clear(self):
        """Clear all cached lattices."""
        self._cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict:
        """Return cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "total_requests": total_requests
        }


# =============================================================================
# API Models
# =============================================================================

class TruthRequest(BaseModel):
    """Request body for /v1/truth endpoint."""
    query: str = Field(..., description="The question to process through SCL")
    system_prompt: Optional[str] = Field(
        None, 
        description="Optional system prompt override"
    )
    threshold: float = Field(
        0.75, 
        ge=0.0, 
        le=1.0,
        description="Cosine similarity threshold for Lattice Meet (δ)"
    )
    use_cache: bool = Field(
        True,
        description="Whether to use/update the Lattice Cache"
    )
    max_tokens: int = Field(
        1024,
        ge=1,
        le=4096,
        description="Maximum tokens per model response"
    )


class TruthResponse(BaseModel):
    """Response body for /v1/truth endpoint."""
    query: str
    response: str
    surviving_atoms: int
    total_atoms: int
    consensus_rate: float
    cached: bool
    processing_time_ms: float
    model_outputs: Optional[Dict[str, str]] = None
    filtered: Optional[Dict[str, List[str]]] = None


class CacheStatsResponse(BaseModel):
    """Response body for /v1/cache/stats endpoint."""
    size: int
    max_size: int
    hits: int
    misses: int
    hit_rate: str
    total_requests: int


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    models_loaded: int
    models: List[str]
    gpu_available: bool
    cache_size: int
    uptime_seconds: float


# =============================================================================
# Global State
# =============================================================================

# These will be initialized in the lifespan context
engine: Optional[SCLEngine] = None
cache: Optional[LatticeCache] = None
start_time: float = 0


def get_default_config() -> SCLConfig:
    """Create default SCL configuration from environment or defaults."""
    models_str = os.environ.get(
        "SCL_MODELS", 
        "llama3.1:8b,mistral:7b,qwen2:7b"
    )
    model_names = [m.strip() for m in models_str.split(",")]
    
    return SCLConfig(
        models=[
            ModelConfig(name=name, persona=f"Model-{i+1}")
            for i, name in enumerate(model_names)
        ],
        similarity_threshold=float(os.environ.get("SCL_THRESHOLD", "0.75")),
        enable_citation_shell=os.environ.get("SCL_CITATION_SHELL", "true").lower() == "true",
        enable_safety_shell=os.environ.get("SCL_SAFETY_SHELL", "true").lower() == "true",
        enable_syntax_shell=os.environ.get("SCL_SYNTAX_SHELL", "true").lower() == "true",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads models on startup, keeps them in VRAM.
    """
    global engine, cache, start_time
    
    logger.info("=" * 60)
    logger.info("SCL-Serve Starting Up")
    logger.info("=" * 60)
    
    # Initialize cache
    cache_size = int(os.environ.get("SCL_CACHE_SIZE", "1000"))
    cache = LatticeCache(max_size=cache_size)
    logger.info(f"Lattice Cache initialized (max_size={cache_size})")
    
    # Initialize SCL Engine
    config = get_default_config()
    logger.info(f"Loading {len(config.models)} models...")
    
    engine = SCLEngine(config)
    engine.initialize()
    
    start_time = time.time()
    logger.info("SCL-Serve Ready!")
    logger.info("=" * 60)
    
    yield
    
    # Cleanup on shutdown
    logger.info("SCL-Serve Shutting Down...")
    if engine:
        engine.shutdown()
    logger.info("Goodbye!")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="SCL Truth API",
    description="""
    Semantic Compression Lattice (SCL) Truth API
    
    A hallucination-free inference engine implementing the Lattice Meet (⊓) 
    operation across a multi-model ensemble.
    
    Based on: "A Rigorous Formalization of the Semantic Compression Lattice"
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global engine, cache, start_time
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Check if GPU is available
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    models_loaded = len(engine.ensemble.models) if engine.ensemble else 0
    model_names = list(engine.ensemble.models.keys()) if engine.ensemble else []
    
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        models=model_names,
        gpu_available=gpu_available,
        cache_size=len(cache._cache) if cache else 0,
        uptime_seconds=time.time() - start_time
    )


@app.post("/v1/truth", response_model=TruthResponse)
async def query_truth(request: TruthRequest):
    """
    Query the SCL Truth Engine.
    
    Computes the Lattice Meet (⊓) across the model ensemble to find
    consensus truth. Results are cached for efficient reuse.
    """
    global engine, cache
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    start = time.time()
    
    # Check cache first
    if request.use_cache and cache:
        cached = cache.get(request.query, request.threshold)
        if cached:
            logger.info(f"Cache HIT for query: {request.query[:50]}...")
            return TruthResponse(
                query=request.query,
                response=cached.response,
                surviving_atoms=cached.surviving_atoms,
                total_atoms=cached.total_atoms,
                consensus_rate=cached.surviving_atoms / cached.total_atoms if cached.total_atoms > 0 else 0,
                cached=True,
                processing_time_ms=(time.time() - start) * 1000,
                model_outputs=cached.model_outputs,
                filtered=cached.filtered
            )
    
    # Cache miss - compute lattice
    logger.info(f"Cache MISS - Computing lattice for: {request.query[:50]}...")
    
    # Temporarily update config threshold if different
    original_threshold = engine.config.similarity_threshold
    engine.config.similarity_threshold = request.threshold
    if engine.lattice_meet:
        engine.lattice_meet.similarity_threshold = request.threshold
    
    try:
        response, metadata = engine.query(
            prompt=request.query,
            system_prompt=request.system_prompt,
            return_metadata=True
        )
    finally:
        # Restore original threshold
        engine.config.similarity_threshold = original_threshold
        if engine.lattice_meet:
            engine.lattice_meet.similarity_threshold = original_threshold
    
    # Store in cache
    if request.use_cache and cache:
        cache.put(request.query, request.threshold, response, metadata)
    
    processing_time = (time.time() - start) * 1000
    
    total_atoms = metadata.get("total_atoms", 0)
    surviving = metadata.get("surviving", 0)
    
    return TruthResponse(
        query=request.query,
        response=response,
        surviving_atoms=surviving,
        total_atoms=total_atoms,
        consensus_rate=surviving / total_atoms if total_atoms > 0 else 0,
        cached=False,
        processing_time_ms=processing_time,
        model_outputs=metadata.get("model_outputs"),
        filtered=metadata.get("filtered")
    )


@app.get("/v1/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Get Lattice Cache statistics."""
    global cache
    
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    stats = cache.stats()
    return CacheStatsResponse(**stats)


@app.delete("/v1/cache")
async def clear_cache():
    """Clear the Lattice Cache."""
    global cache
    
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    cache.clear()
    logger.info("Lattice Cache cleared")
    
    return {"status": "ok", "message": "Cache cleared"}


@app.get("/v1/cache/entries")
async def list_cache_entries():
    """List all cached lattice entries."""
    global cache
    
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    entries = []
    for key, cached in cache._cache.items():
        entries.append({
            "hash": cached.query_hash,
            "query": cached.query[:100] + "..." if len(cached.query) > 100 else cached.query,
            "surviving_atoms": cached.surviving_atoms,
            "total_atoms": cached.total_atoms,
            "hit_count": cached.hit_count,
            "timestamp": datetime.fromtimestamp(cached.timestamp).isoformat()
        })
    
    return {"entries": entries, "count": len(entries)}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("SCL_HOST", "0.0.0.0")
    port = int(os.environ.get("SCL_PORT", "8000"))
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     SCL Truth API Server                         ║
║                                                                  ║
║  Semantic Compression Lattice - Hallucination-Free Inference     ║
╚══════════════════════════════════════════════════════════════════╝

Starting server on http://{host}:{port}

Endpoints:
  POST /v1/truth        - Query the SCL engine
  GET  /v1/cache/stats  - View cache statistics  
  GET  /v1/cache/entries - List cached queries
  DELETE /v1/cache      - Clear the cache
  GET  /health          - Health check

Environment Variables:
  SCL_MODELS         - Comma-separated model names (default: llama3.1:8b,mistral:7b,qwen2:7b)
  SCL_THRESHOLD      - Default similarity threshold (default: 0.75)
  SCL_CACHE_SIZE     - Max cache entries (default: 1000)
  SCL_HOST           - Server host (default: 0.0.0.0)
  SCL_PORT           - Server port (default: 8000)
""")
    
    uvicorn.run(app, host=host, port=port)
