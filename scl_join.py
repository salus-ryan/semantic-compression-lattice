#!/usr/bin/env python3
"""
scl_join.py - Lattice Join (⊔) CLI Tool

Implements the Join operation from Theorem 1, Part 2 of the SCL paper.
Merges multiple semantic lattices into a unified super-lattice.

The Join (⊔) is the Least Upper Bound - it performs SYNTHESIS (union)
while the Meet (⊓) performs CONSENSUS (intersection).

Use cases:
- Law Lattice ⊔ Code Lattice → Legal Tech Agent
- Medicine Lattice ⊔ Research Lattice → Medical Research Agent
- Domain A ⊔ Domain B → Cross-Domain Expert

Usage:
    # Join two lattices
    python scl_join.py join law.json code.json -o legal_tech.json
    
    # Join multiple lattices
    python scl_join.py join law.json code.json ethics.json -o agent.json
    
    # Join with custom conflict resolution
    python scl_join.py join a.json b.json --resolution energy --threshold 0.9
    
    # Create a lattice from text
    python scl_join.py create --name "Law" --domain law --text "Contract law..."
    
    # Inspect a lattice
    python scl_join.py inspect lattice.json
    
    # Demo: Create and join sample lattices
    python scl_join.py demo
"""

import os
import sys
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-Join")


def get_embedder():
    """Lazy load the embedding model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )


def cmd_join(args):
    """Join multiple lattice files."""
    from lattice_ops import SemanticLattice, LatticeJoin
    
    if len(args.lattices) < 2:
        print("Error: Need at least 2 lattices to join")
        return 1
    
    # Load all lattices
    lattices = []
    for path in args.lattices:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return 1
        
        logger.info(f"Loading lattice: {path}")
        lattice = SemanticLattice.load(path)
        lattices.append(lattice)
        logger.info(f"  - {lattice.name}: {len(lattice.vertices)} vertices, {len(lattice.edges)} edges")
    
    # Create join operator
    joiner = LatticeJoin(
        similarity_threshold=args.threshold,
        conflict_resolution=args.resolution
    )
    
    # Set domain priorities if provided
    if args.priority:
        priorities = {}
        for p in args.priority:
            domain, priority = p.split(":")
            priorities[domain] = int(priority)
        joiner.set_domain_priority(priorities)
        logger.info(f"Domain priorities: {priorities}")
    
    # Perform join
    logger.info(f"\nComputing Join (⊔) with threshold={args.threshold}, resolution={args.resolution}")
    
    if len(lattices) == 2:
        result, stats = joiner.compute_join(
            lattices[0], lattices[1],
            name=args.name,
            domain=args.domain
        )
    else:
        result, stats = joiner.join_multiple(
            lattices,
            name=args.name,
            domain=args.domain
        )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("JOIN OPERATION COMPLETE")
    print("=" * 60)
    
    if "join_steps" in stats:
        # Multiple lattices
        print(f"\nTotal joins performed: {stats['total_joins']}")
        for step in stats["join_steps"]:
            print(f"\n  Step {step['join_step']}:")
            print(f"    Vertices: {step['vertices_a']} + {step['vertices_b']} → {step['final_vertices']}")
            print(f"    Conflicts found: {step['conflicts_found']}")
            print(f"    Conflicts resolved: {step['conflicts_resolved']}")
    else:
        # Two lattices
        print(f"\nInput lattices:")
        print(f"  A: {stats['vertices_a']} vertices, {stats['edges_a']} edges")
        print(f"  B: {stats['vertices_b']} vertices, {stats['edges_b']} edges")
        print(f"\nConflict resolution:")
        print(f"  Conflicts found: {stats['conflicts_found']}")
        print(f"  Conflicts resolved: {stats['conflicts_resolved']}")
        print(f"  Vertices merged: {stats['vertices_merged']}")
    
    print(f"\nResult lattice: {result.name}")
    print(f"  Domain: {result.domain}")
    print(f"  Vertices: {len(result.vertices)}")
    print(f"  Edges: {len(result.edges)}")
    
    # Save result
    output_path = args.output or f"{result.name.replace(' ', '_').lower()}.json"
    result.save(output_path)
    print(f"\nSaved to: {output_path}")
    
    return 0


def cmd_create(args):
    """Create a new lattice from text or file."""
    from lattice_ops import (
        SemanticLattice, LatticeVertex, LatticeEdge,
        MeaningAtom, create_lattice_from_atoms
    )
    
    # Get text content
    if args.text:
        content = args.text
    elif args.file:
        with open(args.file, 'r') as f:
            content = f.read()
    else:
        print("Error: Provide --text or --file")
        return 1
    
    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        print("Error: No valid sentences found in content")
        return 1
    
    logger.info(f"Found {len(sentences)} sentences")
    
    # Create atoms
    atoms = []
    for i, sent in enumerate(sentences):
        atom = MeaningAtom(
            text=sent,
            source_model="user_input",
            log_prob=-0.5,  # Default confidence
            semantic_energy=0.5
        )
        atoms.append(atom)
    
    # Embed if requested
    embedder = None
    if not args.no_embed:
        logger.info("Computing embeddings...")
        embedder = get_embedder()
    
    # Create lattice
    lattice = create_lattice_from_atoms(
        atoms=atoms,
        name=args.name,
        domain=args.domain,
        embedder=embedder
    )
    
    # Add metadata
    lattice.metadata["created_at"] = time.time()
    lattice.metadata["source"] = args.file or "text_input"
    
    # Save
    output_path = args.output or f"{args.name.replace(' ', '_').lower()}.json"
    lattice.save(output_path)
    
    print(f"\nCreated lattice: {args.name}")
    print(f"  Domain: {args.domain}")
    print(f"  Vertices: {len(lattice.vertices)}")
    print(f"  Edges: {len(lattice.edges)}")
    print(f"  Saved to: {output_path}")
    
    return 0


def cmd_inspect(args):
    """Inspect a lattice file."""
    from lattice_ops import SemanticLattice
    
    if not os.path.exists(args.lattice):
        print(f"Error: File not found: {args.lattice}")
        return 1
    
    lattice = SemanticLattice.load(args.lattice)
    
    print("\n" + "=" * 60)
    print(f"LATTICE: {lattice.name}")
    print("=" * 60)
    print(f"Domain: {lattice.domain}")
    print(f"Vertices: {len(lattice.vertices)}")
    print(f"Edges: {len(lattice.edges)}")
    
    if lattice.metadata:
        print(f"\nMetadata:")
        for k, v in lattice.metadata.items():
            print(f"  {k}: {v}")
    
    # Show vertices
    if args.verbose or args.vertices:
        print(f"\n{'─' * 60}")
        print("VERTICES")
        print(f"{'─' * 60}")
        
        # Sort by energy
        sorted_vertices = sorted(
            lattice.vertices.values(),
            key=lambda v: v.semantic_energy
        )
        
        for v in sorted_vertices[:args.limit]:
            energy_bar = "█" * int(min(v.semantic_energy * 10, 10))
            print(f"\n[{v.id[:8]}] κ={v.semantic_energy:.3f} {energy_bar}")
            print(f"  Domain: {v.source_domain}")
            print(f"  Text: {v.text[:100]}{'...' if len(v.text) > 100 else ''}")
            if v.metadata.get("merged_from"):
                print(f"  Merged from: {v.metadata['merged_from']}")
    
    # Show edges
    if args.verbose or args.edges:
        print(f"\n{'─' * 60}")
        print("EDGES")
        print(f"{'─' * 60}")
        
        for e in list(lattice.edges.values())[:args.limit]:
            print(f"\n[{e.id[:8]}] {e.edge_type} (w={e.weight:.2f})")
            print(f"  {e.source_id[:8]} → {[t[:8] for t in e.target_ids]}")
    
    # Energy distribution
    if lattice.vertices:
        energies = [v.semantic_energy for v in lattice.vertices.values()]
        print(f"\n{'─' * 60}")
        print("ENERGY DISTRIBUTION")
        print(f"{'─' * 60}")
        print(f"  Min: {min(energies):.4f}")
        print(f"  Max: {max(energies):.4f}")
        print(f"  Mean: {np.mean(energies):.4f}")
        print(f"  Std: {np.std(energies):.4f}")
    
    # Domain distribution
    domains = {}
    for v in lattice.vertices.values():
        d = v.source_domain or "unknown"
        domains[d] = domains.get(d, 0) + 1
    
    if domains:
        print(f"\n{'─' * 60}")
        print("DOMAIN DISTRIBUTION")
        print(f"{'─' * 60}")
        for d, count in sorted(domains.items(), key=lambda x: -x[1]):
            pct = count / len(lattice.vertices) * 100
            bar = "█" * int(pct / 5)
            print(f"  {d}: {count} ({pct:.1f}%) {bar}")
    
    return 0


def cmd_demo():
    """Demonstrate the Join operation with sample lattices."""
    from lattice_ops import (
        SemanticLattice, LatticeVertex, LatticeEdge, LatticeJoin,
        MeaningAtom, create_lattice_from_atoms
    )
    
    print("=" * 70)
    print("SCL JOIN (⊔) DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how to merge two domain-specific lattices")
    print("into a unified 'super-lattice' for cross-domain reasoning.\n")
    
    # Create sample "Law" lattice
    law_atoms = [
        MeaningAtom(
            text="A contract requires offer, acceptance, and consideration.",
            semantic_energy=0.3,
            log_prob=-0.3
        ),
        MeaningAtom(
            text="Breach of contract occurs when a party fails to perform their obligations.",
            semantic_energy=0.4,
            log_prob=-0.4
        ),
        MeaningAtom(
            text="Damages are the monetary compensation for breach of contract.",
            semantic_energy=0.35,
            log_prob=-0.35
        ),
        MeaningAtom(
            text="Software licenses are a type of contract governing software use.",
            semantic_energy=0.5,  # Higher energy - less certain
            log_prob=-0.5
        ),
    ]
    
    # Create sample "Code" lattice
    code_atoms = [
        MeaningAtom(
            text="Open source licenses define how software can be used and distributed.",
            semantic_energy=0.25,
            log_prob=-0.25
        ),
        MeaningAtom(
            text="The MIT license allows commercial use with attribution.",
            semantic_energy=0.3,
            log_prob=-0.3
        ),
        MeaningAtom(
            text="Software licenses are legal agreements for software usage rights.",
            semantic_energy=0.35,  # Similar to law lattice but different wording
            log_prob=-0.35
        ),
        MeaningAtom(
            text="GPL requires derivative works to also be open source.",
            semantic_energy=0.4,
            log_prob=-0.4
        ),
    ]
    
    print("Creating sample lattices...")
    
    # Load embedder
    embedder = get_embedder()
    
    # Create lattices
    law_lattice = create_lattice_from_atoms(law_atoms, "Law Lattice", "law", embedder)
    code_lattice = create_lattice_from_atoms(code_atoms, "Code Lattice", "code", embedder)
    
    print(f"\n📚 Law Lattice: {len(law_lattice.vertices)} vertices")
    for v in law_lattice.vertices.values():
        print(f"   κ={v.semantic_energy:.2f} | {v.text[:60]}...")
    
    print(f"\n💻 Code Lattice: {len(code_lattice.vertices)} vertices")
    for v in code_lattice.vertices.values():
        print(f"   κ={v.semantic_energy:.2f} | {v.text[:60]}...")
    
    # Perform Join
    print("\n" + "=" * 70)
    print("COMPUTING JOIN: Law ⊔ Code")
    print("=" * 70)
    
    joiner = LatticeJoin(
        similarity_threshold=0.75,  # Lower threshold to catch semantic overlap
        conflict_resolution="energy"
    )
    
    result, stats = joiner.compute_join(
        law_lattice, code_lattice,
        name="Legal Tech Lattice",
        domain="legal_tech"
    )
    
    print(f"\nConflict Resolution Strategy: energy (lower κ wins)")
    print(f"Similarity Threshold: 0.75")
    print(f"\nResults:")
    print(f"  Conflicts found: {stats['conflicts_found']}")
    print(f"  Conflicts resolved: {stats['conflicts_resolved']}")
    print(f"  Vertices merged: {stats['vertices_merged']}")
    print(f"  Final vertices: {stats['final_vertices']}")
    print(f"  Final edges: {stats['final_edges']}")
    
    print("\n" + "=" * 70)
    print("MERGED LATTICE: Legal Tech")
    print("=" * 70)
    
    # Show merged vertices
    for v in sorted(result.vertices.values(), key=lambda x: x.semantic_energy):
        merged_marker = "🔀" if v.metadata.get("merged_from") else "  "
        domain_marker = {"law": "📚", "code": "💻", "legal_tech": "⚖️"}.get(v.source_domain, "  ")
        print(f"\n{merged_marker} {domain_marker} κ={v.semantic_energy:.2f}")
        print(f"   {v.text[:70]}...")
        if v.metadata.get("merged_from"):
            print(f"   └─ Merged (similarity: {v.metadata.get('merge_similarity', 0):.2f})")
            if v.metadata.get("alternate_text"):
                print(f"      Alt: {v.metadata['alternate_text'][:50]}...")
    
    # Save demo lattices
    law_lattice.save("demo_law.json")
    code_lattice.save("demo_code.json")
    result.save("demo_legal_tech.json")
    
    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print("  demo_law.json       - Law domain lattice")
    print("  demo_code.json      - Code domain lattice")
    print("  demo_legal_tech.json - Joined lattice (Law ⊔ Code)")
    print("\nYou can now:")
    print("  1. Inspect: python scl_join.py inspect demo_legal_tech.json -v")
    print("  2. Join more: python scl_join.py join demo_legal_tech.json other.json")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="SCL Lattice Join (⊔) Tool - Merge semantic lattices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Join two lattices
  python scl_join.py join law.json code.json -o legal_tech.json
  
  # Join with energy-based conflict resolution (default)
  python scl_join.py join a.json b.json --resolution energy
  
  # Join with domain priority
  python scl_join.py join a.json b.json --priority verified:10 academic:8 web:3
  
  # Create a lattice from text
  python scl_join.py create --name "My Domain" --domain mydomain --text "..."
  
  # Inspect a lattice
  python scl_join.py inspect lattice.json --verbose
  
  # Run demo
  python scl_join.py demo
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Join command
    join_parser = subparsers.add_parser("join", help="Join multiple lattices")
    join_parser.add_argument("lattices", nargs="+", help="Lattice JSON files to join")
    join_parser.add_argument("-o", "--output", help="Output file path")
    join_parser.add_argument("-n", "--name", help="Name for joined lattice")
    join_parser.add_argument("-d", "--domain", help="Domain for joined lattice")
    join_parser.add_argument(
        "-t", "--threshold", 
        type=float, 
        default=0.85,
        help="Similarity threshold for conflict detection (default: 0.85)"
    )
    join_parser.add_argument(
        "-r", "--resolution",
        choices=["energy", "recency", "source_priority"],
        default="energy",
        help="Conflict resolution strategy (default: energy)"
    )
    join_parser.add_argument(
        "-p", "--priority",
        nargs="*",
        help="Domain priorities for source_priority resolution (format: domain:priority)"
    )
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new lattice")
    create_parser.add_argument("-n", "--name", required=True, help="Lattice name")
    create_parser.add_argument("-d", "--domain", required=True, help="Domain label")
    create_parser.add_argument("--text", help="Text content to convert to lattice")
    create_parser.add_argument("--file", help="File containing text content")
    create_parser.add_argument("-o", "--output", help="Output file path")
    create_parser.add_argument("--no-embed", action="store_true", help="Skip embedding computation")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a lattice file")
    inspect_parser.add_argument("lattice", help="Lattice JSON file to inspect")
    inspect_parser.add_argument("-v", "--verbose", action="store_true", help="Show all details")
    inspect_parser.add_argument("--vertices", action="store_true", help="Show vertices")
    inspect_parser.add_argument("--edges", action="store_true", help="Show edges")
    inspect_parser.add_argument("-l", "--limit", type=int, default=20, help="Max items to show")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    
    args = parser.parse_args()
    
    if args.command == "join":
        return cmd_join(args)
    elif args.command == "create":
        return cmd_create(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "demo":
        return cmd_demo()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
