#!/usr/bin/env python3
"""
scl_visualize.py - Lattice Visualization Tool

Visualizes the Semantic Compression Lattice as an interactive graph:
- Nodes: Verified meaning atoms (facts)
- Edges: Semantic connections between atoms
- Clusters: "Invariant Shells" - high-density truth regions

Uses NetworkX for graph construction and PyVis for interactive HTML visualization.

Usage:
    # Visualize from SCL cache
    python scl_visualize.py --from-cache
    
    # Visualize a specific query's lattice
    python scl_visualize.py --query "Tell me about Einstein"
    
    # Export to various formats
    python scl_visualize.py --from-cache --output lattice.html
    python scl_visualize.py --from-cache --format png --output lattice.png
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-Visualize")


@dataclass
class AtomNode:
    """A node in the lattice graph representing a meaning atom."""
    id: str
    text: str
    source_query: str
    model_count: int  # How many models agreed
    energy: float  # Semantic energy (lower = more stable)
    cluster: Optional[str] = None


@dataclass
class SemanticEdge:
    """An edge representing semantic similarity between atoms."""
    source_id: str
    target_id: str
    similarity: float
    edge_type: str  # "consensus", "related", "derived"


class LatticeGraph:
    """
    Graph representation of the Semantic Compression Lattice.
    
    Builds a graph where:
    - Nodes are verified meaning atoms
    - Edges connect semantically similar atoms
    - Clusters represent "Invariant Shells" of related truth
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.nodes: Dict[str, AtomNode] = {}
        self.edges: List[SemanticEdge] = []
        self._embedder = None
        self._embeddings: Dict[str, np.ndarray] = {}
    
    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
        return self._embedder
    
    def _hash_text(self, text: str) -> str:
        """Create a short hash ID for text."""
        return hashlib.md5(text.encode()).hexdigest()[:8]
    
    def add_atom(
        self,
        text: str,
        source_query: str,
        model_count: int = 3,
        energy: float = -0.5
    ) -> str:
        """
        Add a meaning atom to the graph.
        
        Returns the node ID.
        """
        node_id = self._hash_text(text)
        
        if node_id not in self.nodes:
            self.nodes[node_id] = AtomNode(
                id=node_id,
                text=text,
                source_query=source_query,
                model_count=model_count,
                energy=energy
            )
            
            # Compute embedding
            embedder = self._get_embedder()
            emb = embedder.encode(
                [f"search_document: {text}"],
                normalize_embeddings=True
            )[0]
            self._embeddings[node_id] = emb
        
        return node_id
    
    def compute_edges(self):
        """
        Compute edges between all nodes based on semantic similarity.
        
        Creates edges where similarity > threshold.
        """
        logger.info("Computing semantic edges...")
        
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        
        if n < 2:
            return
        
        # Build embedding matrix
        embeddings = np.array([self._embeddings[nid] for nid in node_ids])
        
        # Compute pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)
        
        # Create edges for pairs above threshold
        self.edges = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(similarities[i, j])
                if sim > self.similarity_threshold:
                    # Determine edge type based on similarity
                    if sim > 0.9:
                        edge_type = "consensus"
                    elif sim > 0.8:
                        edge_type = "related"
                    else:
                        edge_type = "derived"
                    
                    self.edges.append(SemanticEdge(
                        source_id=node_ids[i],
                        target_id=node_ids[j],
                        similarity=sim,
                        edge_type=edge_type
                    ))
        
        logger.info(f"Created {len(self.edges)} edges")
    
    def detect_clusters(self):
        """
        Detect clusters (Invariant Shells) using connected components.
        
        Assigns cluster labels to nodes.
        """
        import networkx as nx
        
        # Build NetworkX graph
        G = nx.Graph()
        
        for node_id, node in self.nodes.items():
            G.add_node(node_id)
        
        for edge in self.edges:
            G.add_edge(edge.source_id, edge.target_id, weight=edge.similarity)
        
        # Find connected components
        components = list(nx.connected_components(G))
        
        # Assign cluster labels
        for i, component in enumerate(components):
            cluster_label = f"Shell-{i+1}"
            for node_id in component:
                self.nodes[node_id].cluster = cluster_label
        
        logger.info(f"Detected {len(components)} Invariant Shells")
        
        return components
    
    def to_networkx(self):
        """Convert to NetworkX graph."""
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            G.add_node(
                node_id,
                label=node.text[:50] + "..." if len(node.text) > 50 else node.text,
                full_text=node.text,
                source_query=node.source_query,
                model_count=node.model_count,
                energy=node.energy,
                cluster=node.cluster or "Unknown"
            )
        
        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                weight=edge.similarity,
                edge_type=edge.edge_type
            )
        
        return G
    
    def visualize_pyvis(
        self,
        output_path: str = "lattice_graph.html",
        height: str = "800px",
        width: str = "100%",
        notebook: bool = False
    ):
        """
        Create interactive visualization using PyVis.
        
        Generates an HTML file with the interactive graph.
        """
        from pyvis.network import Network
        
        logger.info("Generating PyVis visualization...")
        
        # Create PyVis network
        net = Network(
            height=height,
            width=width,
            notebook=notebook,
            bgcolor="#1a1a2e",
            font_color="white"
        )
        
        # Configure physics
        net.barnes_hut(
            gravity=-5000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.01
        )
        
        # Color palette for clusters
        colors = [
            "#00d4ff",  # Cyan
            "#ff6b6b",  # Red
            "#4ecdc4",  # Teal
            "#ffe66d",  # Yellow
            "#95e1d3",  # Mint
            "#f38181",  # Coral
            "#aa96da",  # Purple
            "#fcbad3",  # Pink
        ]
        
        # Map clusters to colors
        clusters = set(n.cluster for n in self.nodes.values() if n.cluster)
        cluster_colors = {
            c: colors[i % len(colors)] 
            for i, c in enumerate(sorted(clusters))
        }
        
        # Add nodes
        for node_id, node in self.nodes.items():
            # Size based on model consensus
            size = 15 + (node.model_count * 5)
            
            # Color based on cluster
            color = cluster_colors.get(node.cluster, "#888888")
            
            # Tooltip with full info
            title = f"""
            <b>Atom:</b> {node.text}<br>
            <b>Source:</b> {node.source_query[:50]}...<br>
            <b>Models:</b> {node.model_count}<br>
            <b>Energy:</b> {node.energy:.3f}<br>
            <b>Cluster:</b> {node.cluster}
            """
            
            net.add_node(
                node_id,
                label=node.text[:30] + "..." if len(node.text) > 30 else node.text,
                title=title,
                size=size,
                color=color,
                font={"size": 10}
            )
        
        # Add edges
        for edge in self.edges:
            # Width based on similarity
            width = 1 + (edge.similarity - self.similarity_threshold) * 10
            
            # Color based on edge type
            if edge.edge_type == "consensus":
                color = "#00ff00"  # Green for strong consensus
            elif edge.edge_type == "related":
                color = "#ffff00"  # Yellow for related
            else:
                color = "#666666"  # Gray for weak
            
            net.add_edge(
                edge.source_id,
                edge.target_id,
                width=width,
                color=color,
                title=f"Similarity: {edge.similarity:.3f}"
            )
        
        # Save to HTML
        net.save_graph(output_path)
        logger.info(f"Visualization saved to {output_path}")
        
        return output_path
    
    def visualize_matplotlib(
        self,
        output_path: str = "lattice_graph.png",
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Create static visualization using matplotlib.
        
        Good for publications and reports.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        logger.info("Generating matplotlib visualization...")
        
        G = self.to_networkx()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Color nodes by cluster
        clusters = set(nx.get_node_attributes(G, 'cluster').values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        cluster_colors = dict(zip(sorted(clusters), colors))
        
        node_colors = [
            cluster_colors.get(G.nodes[n].get('cluster', 'Unknown'), 'gray')
            for n in G.nodes()
        ]
        
        # Size nodes by model count
        node_sizes = [
            200 + G.nodes[n].get('model_count', 1) * 100
            for n in G.nodes()
        ]
        
        # Draw edges
        edge_weights = [G[u][v].get('weight', 0.5) for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#444444',
            width=[w * 2 for w in edge_weights],
            alpha=0.5,
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )
        
        # Draw labels
        labels = {n: G.nodes[n].get('label', n)[:20] for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_color='white',
            ax=ax
        )
        
        # Title
        ax.set_title(
            "Semantic Compression Lattice\nInvariant Shells Visualization",
            fontsize=16,
            color='white',
            pad=20
        )
        
        # Legend
        legend_elements = [
            plt.scatter([], [], c=[cluster_colors[c]], s=100, label=c)
            for c in sorted(clusters)
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            facecolor='#2a2a4e',
            edgecolor='white',
            labelcolor='white'
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        
        return output_path
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        import networkx as nx
        
        G = self.to_networkx()
        
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "clusters": len(set(n.cluster for n in self.nodes.values() if n.cluster)),
            "density": nx.density(G) if len(G) > 1 else 0,
            "avg_clustering": nx.average_clustering(G) if len(G) > 1 else 0
        }


class LatticeVisualizer:
    """
    Main visualizer that connects to SCL API and builds graphs.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        similarity_threshold: float = 0.7
    ):
        self.api_url = api_url.rstrip("/")
        self.similarity_threshold = similarity_threshold
        self.graph = LatticeGraph(similarity_threshold)
    
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
    
    def _query_scl(self, query: str) -> Dict:
        """Query the SCL API and get full response."""
        payload = {
            "query": query,
            "threshold": 0.75,
            "use_cache": True,
            "max_tokens": 1024
        }
        
        response = requests.post(
            f"{self.api_url}/v1/truth",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    
    def _extract_atoms(self, text: str) -> List[str]:
        """Extract atoms (sentences) from text."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def build_from_cache(self):
        """
        Build graph from all cached SCL entries.
        """
        logger.info("Building graph from SCL cache...")
        
        # For now, we need to query each cached entry to get full response
        # This is a limitation - ideally we'd have an API to get full cached data
        
        entries = self._fetch_cache_entries()
        logger.info(f"Found {len(entries)} cached entries")
        
        for entry in entries:
            query = entry.get("query", "")
            surviving = entry.get("surviving_atoms", 0)
            
            if surviving > 0:
                # Re-query to get full response (will be cached)
                try:
                    result = self._query_scl(query)
                    response = result.get("response", "")
                    
                    atoms = self._extract_atoms(response)
                    for atom in atoms:
                        self.graph.add_atom(
                            text=atom,
                            source_query=query,
                            model_count=3,  # Assume consensus
                            energy=-0.5
                        )
                except Exception as e:
                    logger.warning(f"Failed to process entry: {e}")
        
        # Compute edges and clusters
        self.graph.compute_edges()
        self.graph.detect_clusters()
        
        logger.info(f"Graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
    
    def build_from_query(self, query: str):
        """
        Build graph from a single query's results.
        """
        logger.info(f"Building graph from query: {query[:50]}...")
        
        result = self._query_scl(query)
        
        response = result.get("response", "")
        model_outputs = result.get("model_outputs", {})
        
        # Add atoms from final response
        atoms = self._extract_atoms(response)
        for atom in atoms:
            self.graph.add_atom(
                text=atom,
                source_query=query,
                model_count=3,
                energy=-0.5
            )
        
        # Add atoms from individual model outputs (for comparison)
        for model_name, output in model_outputs.items():
            model_atoms = self._extract_atoms(output)
            for atom in model_atoms:
                self.graph.add_atom(
                    text=atom,
                    source_query=f"{query} [{model_name}]",
                    model_count=1,
                    energy=-0.3
                )
        
        # Compute edges and clusters
        self.graph.compute_edges()
        self.graph.detect_clusters()
        
        logger.info(f"Graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
    
    def visualize(
        self,
        output_path: str = "lattice_graph.html",
        format: str = "html"
    ) -> str:
        """
        Generate visualization.
        
        Args:
            output_path: Output file path
            format: "html" for interactive PyVis, "png" for static matplotlib
        
        Returns:
            Path to generated file
        """
        if format == "html":
            return self.graph.visualize_pyvis(output_path)
        elif format == "png":
            return self.graph.visualize_matplotlib(output_path)
        else:
            raise ValueError(f"Unknown format: {format}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SCL Lattice Visualization - See the Truth Graph"
    )
    
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Build graph from all cached SCL entries"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Build graph from a specific query"
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
        default=0.7,
        help="Similarity threshold for edges (default: 0.7)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="lattice_graph.html",
        help="Output file path (default: lattice_graph.html)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["html", "png"],
        default="html",
        help="Output format: html (interactive) or png (static)"
    )
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(
        api_url=args.api_url,
        similarity_threshold=args.threshold
    )
    
    # Check server
    if not visualizer._check_server():
        logger.error(f"SCL server not available at {args.api_url}")
        logger.error("Start the server with: python scl_serve.py")
        return 1
    
    # Build graph
    if args.from_cache:
        visualizer.build_from_cache()
    elif args.query:
        visualizer.build_from_query(args.query)
    else:
        parser.print_help()
        return 0
    
    # Generate visualization
    output_path = visualizer.visualize(args.output, args.format)
    
    # Print stats
    stats = visualizer.graph.get_stats()
    print("\n" + "=" * 60)
    print("LATTICE VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Nodes (atoms):     {stats['nodes']}")
    print(f"Edges:             {stats['edges']}")
    print(f"Invariant Shells:  {stats['clusters']}")
    print(f"Graph density:     {stats['density']:.3f}")
    print(f"Avg clustering:    {stats['avg_clustering']:.3f}")
    print(f"\nOutput: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
