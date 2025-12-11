#!/usr/bin/env python3
"""
scl_steering.py - Activation Steering for SCL

Implements practical approaches to inject steered vectors back into generation:

1. Semantic Prompt Steering - Convert steering vectors to natural language
   prefixes that bias generation toward the target direction.

2. Contrastive Decoding - Generate from steered vs baseline prompts,
   amplify the difference to get "more aligned" outputs.

3. Iterative Refinement - Generate -> Embed -> Flow -> Find nearest candidate

Reference: Definition 9 (Gradient Flow), applied to generation.
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SCL-Steering")


class SemanticPromptSteerer:
    """
    Convert steering vectors to natural language prompt prefixes.
    
    If we can't directly inject vectors into the model, we can find text 
    that EMBEDS to similar vectors and prepend it as context.
    """
    
    def __init__(self):
        self._embedder = None
        self._steering_phrases: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
        return self._embedder
    
    def add_steering_phrases(self, objective: str, phrases: List[str]):
        """Add phrases that represent a steering objective."""
        embedder = self._get_embedder()
        texts = [f"search_document: {p}" for p in phrases]
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        
        self._steering_phrases[objective] = [
            (phrase, emb) for phrase, emb in zip(phrases, embeddings)
        ]
    
    def find_steering_prefix(
        self,
        target_direction: np.ndarray,
        objective: str,
        top_k: int = 3
    ) -> str:
        """Find phrases most aligned with the target steering direction."""
        if objective not in self._steering_phrases:
            return ""
        
        phrases = self._steering_phrases[objective]
        scored = []
        for phrase, emb in phrases:
            score = float(np.dot(emb, target_direction))
            scored.append((score, phrase))
        
        scored.sort(reverse=True)
        top_phrases = [p for _, p in scored[:top_k]]
        return " ".join(top_phrases)
    
    def create_steered_system_prompt(
        self,
        base_prompt: str,
        objectives: Dict[str, float],
        target_directions: Dict[str, np.ndarray]
    ) -> str:
        """Create a system prompt that incorporates steering directions."""
        prefixes = []
        
        for obj_name, weight in objectives.items():
            if obj_name in target_directions and weight > 0:
                prefix = self.find_steering_prefix(
                    target_directions[obj_name],
                    obj_name,
                    top_k=max(1, int(weight * 3))
                )
                if prefix:
                    prefixes.append(prefix)
        
        if prefixes:
            steering_text = " ".join(prefixes)
            return f"{steering_text}\n\n{base_prompt}"
        
        return base_prompt


class ContrastiveDecoder:
    """
    Generate using contrastive decoding between steered and baseline.
    
    Generate two responses - one with steering prompt, one without.
    Compare the difference to measure steering effect.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self._embedder = None
    
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
        return self._embedder
    
    def generate(
        self,
        prompt: str,
        model: str = "llama3.1:8b",
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Generate with Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", ""), []
        except Exception as e:
            logger.error(f"Generation failed: {e}")
        return "", []
    
    def contrastive_generate(
        self,
        prompt: str,
        steered_system: str,
        baseline_system: str,
        model: str = "llama3.1:8b",
        max_tokens: int = 256
    ) -> Tuple[str, str, Dict]:
        """Generate using contrastive decoding."""
        steered_response, _ = self.generate(prompt, model, steered_system, max_tokens)
        baseline_response, _ = self.generate(prompt, model, baseline_system, max_tokens)
        
        embedder = self._get_embedder()
        
        if steered_response and baseline_response:
            emb_steered = embedder.encode(
                [f"search_document: {steered_response}"],
                normalize_embeddings=True
            )[0]
            emb_baseline = embedder.encode(
                [f"search_document: {baseline_response}"],
                normalize_embeddings=True
            )[0]
            
            similarity = float(np.dot(emb_steered, emb_baseline))
            difference_norm = float(np.linalg.norm(emb_steered - emb_baseline))
        else:
            similarity = 0
            difference_norm = 0
        
        metrics = {
            "similarity": similarity,
            "difference_norm": difference_norm,
        }
        
        return steered_response, baseline_response, metrics


class IterativeRefinementSteerer:
    """
    Iteratively refine generation toward steering objectives.
    
    Algorithm:
    1. Generate initial response
    2. Embed response
    3. Apply gradient flow to embedding
    4. Find candidate closest to flowed embedding
    5. Repeat until convergence
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        max_iterations: int = 3,
        embedder=None
    ):
        self.ollama_url = ollama_url
        self.max_iterations = max_iterations
        self._embedder = embedder
        self._dynamics_engine = None
    
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True,
                device="cpu"  # Use CPU to avoid GPU OOM
            )
        return self._embedder
    
    def _get_dynamics_engine(self):
        if self._dynamics_engine is None:
            from scl_dynamics import SCLDynamicsEngine
            self._dynamics_engine = SCLDynamicsEngine()
        return self._dynamics_engine
    
    def generate_candidates(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        n_candidates: int = 3,
        temperature: float = 0.9
    ) -> List[str]:
        """Generate multiple candidate responses."""
        candidates = []
        
        for _ in range(n_candidates):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                        "stream": False,
                        "options": {
                            "num_predict": 256,
                            "temperature": temperature,
                        }
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    text = response.json().get("response", "")
                    if text:
                        candidates.append(text)
            except Exception as e:
                logger.warning(f"Candidate generation failed: {e}")
        
        return candidates
    
    def select_best_candidate(
        self,
        candidates: List[str],
        target_embedding: np.ndarray
    ) -> Tuple[str, float]:
        """Select candidate closest to target embedding."""
        if not candidates:
            return "", 0.0
        
        embedder = self._get_embedder()
        texts = [f"search_document: {c}" for c in candidates]
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        
        best_idx = 0
        best_score = float('-inf')
        
        for i, emb in enumerate(embeddings):
            score = float(np.dot(emb, target_embedding))
            if score > best_score:
                best_score = score
                best_idx = i
        
        return candidates[best_idx], best_score
    
    def steer_generation(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str = "llama3.1:8b",
        alignment_objectives: Optional[Dict[str, float]] = None
    ) -> Tuple[str, Dict]:
        """Generate with iterative steering toward alignment objectives."""
        engine = self._get_dynamics_engine()
        embedder = self._get_embedder()
        
        # Add objectives if provided
        if alignment_objectives:
            for obj_name, weight in alignment_objectives.items():
                if obj_name not in engine.teleological.objectives:
                    if obj_name == "helpful":
                        engine.add_alignment_objective(
                            "helpful",
                            ["I'd be happy to help.", "Here's the solution."],
                            ["I can't help.", "Figure it out yourself."],
                            weight=weight
                        )
                    elif obj_name == "safe":
                        engine.add_alignment_objective(
                            "safe",
                            ["Please be careful.", "Consult a professional."],
                            ["Don't worry about safety.", "Just do it."],
                            weight=weight
                        )
        
        iteration_info = {
            "iterations": [],
            "initial_response": "",
            "final_response": ""
        }
        
        current_response = ""
        current_system = system_prompt
        
        for iteration in range(self.max_iterations):
            candidates = self.generate_candidates(
                prompt, current_system, model,
                n_candidates=3 if iteration == 0 else 2,
                temperature=0.8 - (iteration * 0.1)
            )
            
            if not candidates:
                break
            
            if iteration == 0:
                iteration_info["initial_response"] = candidates[0]
            
            texts = [f"search_document: {c}" for c in candidates]
            embeddings = embedder.encode(texts, normalize_embeddings=True)
            
            best_embedding = embeddings[0]
            
            flowed_embedding, flow_info = engine.correct_vector(
                best_embedding,
                semantic_energy=0.5,
                return_trajectory=True
            )
            
            best_candidate, alignment_score = self.select_best_candidate(
                candidates, flowed_embedding
            )
            
            iter_data = {
                "iteration": iteration,
                "n_candidates": len(candidates),
                "alignment_score": alignment_score,
                "energy_reduction": flow_info["energy_reduction"],
                "selected": best_candidate[:100] + "..."
            }
            iteration_info["iterations"].append(iter_data)
            
            current_response = best_candidate
            
            if iteration < self.max_iterations - 1:
                current_system = f"{system_prompt}\n\nPrevious good response style: {best_candidate[:200]}"
        
        iteration_info["final_response"] = current_response
        
        return current_response, iteration_info


class SCLSteeringEngine:
    """
    Unified steering engine combining all approaches.
    
    Usage:
        engine = SCLSteeringEngine()
        engine.add_objective("helpful", positive_examples, negative_examples)
        
        response = engine.generate_steered(
            prompt="Explain quantum physics",
            objectives={"helpful": 1.0, "concise": 0.5}
        )
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.prompt_steerer = SemanticPromptSteerer()
        self.contrastive = ContrastiveDecoder(ollama_url)
        self.iterative = IterativeRefinementSteerer(ollama_url)
        self._dynamics_engine = None
        
        self._init_default_phrases()
    
    def _init_default_phrases(self):
        """Initialize default steering phrases for common objectives."""
        self.prompt_steerer.add_steering_phrases("helpful", [
            "I want to be as helpful as possible.",
            "Let me provide a clear and useful answer.",
            "I'll do my best to assist you.",
            "Here's a thorough explanation.",
        ])
        
        self.prompt_steerer.add_steering_phrases("safe", [
            "Safety is my top priority.",
            "I'll provide responsible guidance.",
            "Please exercise caution.",
            "Consult professionals when needed.",
        ])
        
        self.prompt_steerer.add_steering_phrases("concise", [
            "I'll be brief and to the point.",
            "Here's the short answer.",
            "In summary:",
            "The key point is:",
        ])
    
    def _get_dynamics_engine(self):
        if self._dynamics_engine is None:
            from scl_dynamics import SCLDynamicsEngine
            self._dynamics_engine = SCLDynamicsEngine()
        return self._dynamics_engine
    
    def add_objective(
        self,
        name: str,
        positive_examples: List[str],
        negative_examples: List[str],
        weight: float = 1.0
    ):
        """Add a custom alignment objective."""
        engine = self._get_dynamics_engine()
        engine.add_alignment_objective(
            name, positive_examples, negative_examples, weight
        )
        self.prompt_steerer.add_steering_phrases(name, positive_examples)
    
    def generate_steered(
        self,
        prompt: str,
        objectives: Dict[str, float],
        model: str = "llama3.1:8b",
        method: str = "iterative",
        base_system: str = "You are a helpful assistant.",
        max_tokens: int = 256
    ) -> Tuple[str, Dict]:
        """
        Generate a response steered toward the specified objectives.
        
        Args:
            prompt: User prompt
            objectives: Dict of objective name -> weight
            model: Ollama model name
            method: "prompt", "contrastive", or "iterative"
            base_system: Base system prompt
            max_tokens: Maximum tokens to generate
        
        Returns:
            (response, metadata)
        """
        engine = self._get_dynamics_engine()
        
        if method == "prompt":
            target_directions = {}
            for obj_name in objectives:
                if obj_name in engine.teleological.objectives:
                    target_directions[obj_name] = engine.teleological.objectives[obj_name].direction
            
            steered_system = self.prompt_steerer.create_steered_system_prompt(
                base_system, objectives, target_directions
            )
            
            response, _ = self.contrastive.generate(
                prompt, model, steered_system, max_tokens
            )
            
            return response, {"method": "prompt", "system": steered_system}
        
        elif method == "contrastive":
            target_directions = {}
            for obj_name in objectives:
                if obj_name in engine.teleological.objectives:
                    target_directions[obj_name] = engine.teleological.objectives[obj_name].direction
            
            steered_system = self.prompt_steerer.create_steered_system_prompt(
                base_system, objectives, target_directions
            )
            
            steered, baseline, metrics = self.contrastive.contrastive_generate(
                prompt, steered_system, base_system, model
            )
            
            return steered, {
                "method": "contrastive",
                "baseline": baseline,
                "metrics": metrics
            }
        
        elif method == "iterative":
            response, info = self.iterative.steer_generation(
                prompt, base_system, model, objectives
            )
            
            return response, {"method": "iterative", "info": info}
        
        else:
            raise ValueError(f"Unknown method: {method}")


def demo_steering():
    """Demonstrate steering approaches."""
    print("=" * 70)
    print("SCL STEERING DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how to inject steered vectors back into generation.")
    print("We'll compare baseline vs steered outputs.\n")
    
    engine = SCLSteeringEngine()
    
    engine.add_objective(
        "educational",
        [
            "Let me explain this concept step by step.",
            "The key principle here is...",
            "To understand this, first consider...",
            "Here's an analogy that might help:",
        ],
        [
            "It's complicated, just trust me.",
            "You wouldn't understand.",
            "That's just how it works.",
            "Don't worry about the details.",
        ],
        weight=1.0
    )
    
    prompt = "Explain why the sky is blue."
    objectives = {"helpful": 1.0, "educational": 1.5, "concise": 0.5}
    
    print(f"Prompt: \"{prompt}\"")
    print(f"Objectives: {objectives}")
    print("\n" + "-" * 70)
    
    print("\n[1] PROMPT STEERING")
    print("-" * 40)
    
    response, metadata = engine.generate_steered(
        prompt, objectives, method="prompt"
    )
    
    print(f"Steered System Prompt:\n{metadata.get('system', '')[:200]}...")
    print(f"\nResponse:\n{response[:500]}...")
    
    print("\n" + "-" * 70)
    print("\n[2] CONTRASTIVE DECODING")
    print("-" * 40)
    
    response, metadata = engine.generate_steered(
        prompt, objectives, method="contrastive"
    )
    
    print(f"Steered Response:\n{response[:300]}...")
    print(f"\nBaseline Response:\n{metadata.get('baseline', '')[:300]}...")
    print(f"\nSimilarity: {metadata.get('metrics', {}).get('similarity', 0):.3f}")
    
    print("\n" + "-" * 70)
    print("\n[3] ITERATIVE REFINEMENT")
    print("-" * 40)
    
    response, metadata = engine.generate_steered(
        prompt, objectives, method="iterative"
    )
    
    info = metadata.get("info", {})
    print(f"Iterations: {len(info.get('iterations', []))}")
    
    if info.get("iterations"):
        for iter_data in info["iterations"]:
            print(f"  Step {iter_data['iteration']}: "
                  f"alignment={iter_data['alignment_score']:.3f}, "
                  f"energy_reduction={iter_data['energy_reduction']:.3f}")
    
    print(f"\nInitial Response:\n{info.get('initial_response', '')[:200]}...")
    print(f"\nFinal Response:\n{response[:300]}...")
    
    print("\n" + "=" * 70)
    print("STEERING COMPLETE")
    print("=" * 70)
    
    return engine


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SCL Steering - Inject steered vectors into generation"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--prompt", type=str, help="Prompt to generate from")
    parser.add_argument("--method", choices=["prompt", "contrastive", "iterative"],
                       default="iterative", help="Steering method")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model to use")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_steering()
    elif args.prompt:
        engine = SCLSteeringEngine()
        response, metadata = engine.generate_steered(
            args.prompt,
            {"helpful": 1.0},
            model=args.model,
            method=args.method
        )
        print(f"Response:\n{response}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
