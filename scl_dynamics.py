#!/usr/bin/env python3
"""
scl_dynamics.py - Continuous Dynamics and Alignment for SCL

Implements the remaining theoretical components from the SCL paper:

1. Riemannian Gradient Flow (Φₜ) - Definition 9
   Continuous steering of thought vectors toward truth manifold.
   d/dt Φₜ(v) = -Proj_{T_Φₜ(v) M_I}(∇_v J)
   
2. Teleological Gradient (∇SAL) - Definition 8  
   Purpose vector field for goal-directed alignment.
   J(V) = ... - α⟨∇SAL(v), v⟩

These enable:
- Real-time hallucination correction (not just rejection sampling)
- Goal-directed generation (helpfulness, safety, brevity as vector fields)
- Continuous optimization on the truth manifold

Reference: "A Rigorous Formalization of the Semantic Compression Lattice"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SCL-Dynamics")


# =============================================================================
# Teleological Gradient (∇SAL) - Definition 8
# =============================================================================

@dataclass
class AlignmentObjective:
    """
    A single alignment objective contributing to ∇SAL.
    
    Each objective defines a direction in embedding space that the system
    should move toward (positive weight) or away from (negative weight).
    """
    name: str
    direction: np.ndarray  # Unit vector in embedding space
    weight: float = 1.0  # α coefficient
    description: str = ""
    
    def compute_gradient(self, v: np.ndarray) -> np.ndarray:
        """
        Compute gradient contribution: α * direction
        
        The gradient points in the direction that increases alignment
        with this objective.
        """
        return self.weight * self.direction


class TeleologicalGradient:
    """
    Implements the Teleological Gradient ∇SAL from Definition 8.
    
    The teleological gradient represents the "purpose vector" - a force
    pulling the system toward specific goals beyond just truth/stability.
    
    J(V) = Σ κ(v) + λ‖v‖² - α⟨∇SAL(v), v⟩
    
    The term -α⟨∇SAL(v), v⟩ rewards vectors aligned with the goal direction.
    
    Example objectives:
    - Helpfulness: Move toward helpful response patterns
    - Safety: Move away from harmful content patterns  
    - Brevity: Move toward concise expression patterns
    - Formality: Move toward formal language patterns
    
    These can be learned from examples or defined manually.
    """
    
    def __init__(self, embedding_dim: int = 768):
        """
        Args:
            embedding_dim: Dimension of the embedding space
        """
        self.embedding_dim = embedding_dim
        self.objectives: Dict[str, AlignmentObjective] = {}
        self._embedder = None
    
    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
        return self._embedder
    
    def add_objective(self, objective: AlignmentObjective):
        """Add an alignment objective."""
        # Normalize direction
        norm = np.linalg.norm(objective.direction)
        if norm > 0:
            objective.direction = objective.direction / norm
        self.objectives[objective.name] = objective
    
    def add_objective_from_examples(
        self,
        name: str,
        positive_examples: List[str],
        negative_examples: List[str],
        weight: float = 1.0,
        description: str = ""
    ):
        """
        Learn an alignment direction from positive/negative examples.
        
        The direction is computed as:
        direction = mean(embed(positive)) - mean(embed(negative))
        
        This creates a vector pointing from "bad" toward "good".
        
        Args:
            name: Name for this objective
            positive_examples: Examples of desired behavior
            negative_examples: Examples of undesired behavior
            weight: Importance weight (α)
            description: Human-readable description
        """
        embedder = self._get_embedder()
        
        # Embed examples
        pos_texts = [f"search_document: {t}" for t in positive_examples]
        neg_texts = [f"search_document: {t}" for t in negative_examples]
        
        pos_embeddings = embedder.encode(pos_texts, normalize_embeddings=True)
        neg_embeddings = embedder.encode(neg_texts, normalize_embeddings=True)
        
        # Compute direction: positive centroid - negative centroid
        pos_centroid = np.mean(pos_embeddings, axis=0)
        neg_centroid = np.mean(neg_embeddings, axis=0)
        direction = pos_centroid - neg_centroid
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        objective = AlignmentObjective(
            name=name,
            direction=direction,
            weight=weight,
            description=description
        )
        self.objectives[name] = objective
        
        logger.info(f"Added objective '{name}' from {len(positive_examples)} pos / {len(negative_examples)} neg examples")
    
    def compute_gradient(self, v: np.ndarray) -> np.ndarray:
        """
        Compute the full teleological gradient ∇SAL at point v.
        
        ∇SAL(v) = Σᵢ αᵢ * dᵢ
        
        where dᵢ is the direction for objective i and αᵢ is its weight.
        """
        if not self.objectives:
            return np.zeros_like(v)
        
        gradient = np.zeros_like(v)
        for obj in self.objectives.values():
            gradient += obj.compute_gradient(v)
        
        return gradient
    
    def compute_alignment_score(self, v: np.ndarray) -> Dict[str, float]:
        """
        Compute alignment scores for each objective.
        
        Score = ⟨v, direction⟩ (cosine similarity since both normalized)
        
        Returns dict mapping objective name to alignment score [-1, 1].
        """
        scores = {}
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        
        for name, obj in self.objectives.items():
            score = float(np.dot(v_norm, obj.direction))
            scores[name] = score
        
        return scores
    
    def compute_total_alignment(self, v: np.ndarray) -> float:
        """
        Compute weighted total alignment: Σᵢ αᵢ⟨v, dᵢ⟩
        
        This is the term that appears in the objective functional J(V).
        """
        scores = self.compute_alignment_score(v)
        total = 0.0
        for name, score in scores.items():
            total += self.objectives[name].weight * score
        return total
    
    def save(self, path: str):
        """Save objectives to file."""
        import json
        data = {
            "embedding_dim": self.embedding_dim,
            "objectives": {
                name: {
                    "name": obj.name,
                    "direction": obj.direction.tolist(),
                    "weight": obj.weight,
                    "description": obj.description
                }
                for name, obj in self.objectives.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TeleologicalGradient":
        """Load objectives from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        tg = cls(embedding_dim=data["embedding_dim"])
        for name, obj_data in data["objectives"].items():
            obj = AlignmentObjective(
                name=obj_data["name"],
                direction=np.array(obj_data["direction"], dtype=np.float32),
                weight=obj_data["weight"],
                description=obj_data.get("description", "")
            )
            tg.objectives[name] = obj
        
        return tg


# =============================================================================
# Riemannian Gradient Flow (Φₜ) - Definition 9
# =============================================================================

class InvariantManifold(ABC):
    """
    Abstract base class for invariant manifolds M_I.
    
    An invariant manifold defines the "truth surface" that vectors
    should flow toward during gradient descent.
    """
    
    @abstractmethod
    def project(self, v: np.ndarray) -> np.ndarray:
        """Project vector onto the manifold."""
        pass
    
    @abstractmethod
    def tangent_projection(self, v: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Project gradient onto tangent space at v."""
        pass
    
    @abstractmethod
    def distance(self, v: np.ndarray) -> float:
        """Compute distance from v to manifold."""
        pass


class SphericalManifold(InvariantManifold):
    """
    Spherical manifold: M = {v : ‖v‖ = r}
    
    Embedding vectors are typically normalized, so they lie on a hypersphere.
    This manifold preserves that constraint during gradient flow.
    """
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def project(self, v: np.ndarray) -> np.ndarray:
        """Project onto sphere: v → r * v/‖v‖"""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return self.radius * v / norm
    
    def tangent_projection(self, v: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Project gradient onto tangent space of sphere at v.
        
        T_v S^n = {w : ⟨w, v⟩ = 0}
        
        Proj_T(g) = g - ⟨g, v̂⟩v̂
        """
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        return gradient - np.dot(gradient, v_norm) * v_norm
    
    def distance(self, v: np.ndarray) -> float:
        """Distance to sphere surface."""
        return abs(np.linalg.norm(v) - self.radius)


class ConvexHullManifold(InvariantManifold):
    """
    Convex hull manifold: M = conv({v₁, ..., vₖ})
    
    The truth manifold is the convex hull of verified "truth anchors".
    Vectors flow toward this hull during gradient descent.
    """
    
    def __init__(self, anchors: np.ndarray):
        """
        Args:
            anchors: Array of shape (k, d) containing k anchor vectors
        """
        self.anchors = anchors
        self._centroid = np.mean(anchors, axis=0)
    
    def project(self, v: np.ndarray) -> np.ndarray:
        """
        Project onto convex hull.
        
        Simplified: project toward centroid if outside hull.
        Full implementation would use quadratic programming.
        """
        # Check if inside hull (simplified: check distance to centroid)
        dist_to_centroid = np.linalg.norm(v - self._centroid)
        max_anchor_dist = max(np.linalg.norm(a - self._centroid) for a in self.anchors)
        
        if dist_to_centroid <= max_anchor_dist:
            return v  # Already inside (approximately)
        
        # Project toward centroid
        direction = self._centroid - v
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        return v + direction * (dist_to_centroid - max_anchor_dist)
    
    def tangent_projection(self, v: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Project gradient onto tangent space (identity for interior points)."""
        # For interior points, tangent space is full space
        # For boundary points, would need to compute face normal
        return gradient
    
    def distance(self, v: np.ndarray) -> float:
        """Distance to convex hull (simplified)."""
        dist_to_centroid = np.linalg.norm(v - self._centroid)
        max_anchor_dist = max(np.linalg.norm(a - self._centroid) for a in self.anchors)
        return max(0, dist_to_centroid - max_anchor_dist)


class GradientFlow:
    """
    Implements Riemannian Gradient Flow Φₜ from Definition 9.
    
    The flow equation:
    d/dt Φₜ(v) = -Proj_{T_Φₜ(v) M_I}(∇_v J)
    
    This continuously steers vectors "downhill" on the energy landscape
    while staying on the invariant manifold (truth surface).
    
    The objective functional J(V) combines:
    - Semantic energy κ(v): Stability/confidence
    - Regularization λ‖v‖²: Prevent drift
    - Alignment -α⟨∇SAL, v⟩: Goal-directed steering
    
    Unlike rejection sampling (generate → check → delete), this enables
    real-time correction during generation.
    """
    
    def __init__(
        self,
        manifold: InvariantManifold,
        teleological_gradient: Optional[TeleologicalGradient] = None,
        lambda_reg: float = 0.01,
        learning_rate: float = 0.1,
        max_steps: int = 100,
        convergence_threshold: float = 1e-6
    ):
        """
        Args:
            manifold: The invariant manifold M_I to flow on
            teleological_gradient: Optional ∇SAL for goal-directed flow
            lambda_reg: Regularization weight λ
            learning_rate: Step size for discrete flow
            max_steps: Maximum flow steps
            convergence_threshold: Stop when gradient norm below this
        """
        self.manifold = manifold
        self.teleological_gradient = teleological_gradient
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
    
    def compute_energy_gradient(
        self,
        v: np.ndarray,
        semantic_energy: float,
        energy_gradient: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute ∇J(v) = ∇κ(v) + λv - α∇SAL(v)
        
        Args:
            v: Current vector
            semantic_energy: κ(v) value
            energy_gradient: ∇κ(v) if available, else estimated
        
        Returns:
            Full gradient of objective functional
        """
        # Energy gradient (if not provided, use v direction as proxy)
        if energy_gradient is None:
            # Approximate: gradient points away from origin for high energy
            energy_gradient = semantic_energy * v / (np.linalg.norm(v) + 1e-8)
        
        # Regularization gradient: ∇(λ‖v‖²) = 2λv
        reg_gradient = 2 * self.lambda_reg * v
        
        # Teleological gradient (negative because we want to maximize alignment)
        if self.teleological_gradient is not None:
            sal_gradient = -self.teleological_gradient.compute_gradient(v)
        else:
            sal_gradient = np.zeros_like(v)
        
        return energy_gradient + reg_gradient + sal_gradient
    
    def flow_step(
        self,
        v: np.ndarray,
        semantic_energy: float,
        energy_gradient: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Perform one step of gradient flow.
        
        v_{t+1} = Proj_M(v_t - η * Proj_T(∇J))
        
        Returns:
            (new_v, gradient_norm)
        """
        # Compute full gradient
        gradient = self.compute_energy_gradient(v, semantic_energy, energy_gradient)
        
        # Project onto tangent space of manifold
        tangent_gradient = self.manifold.tangent_projection(v, gradient)
        
        # Take gradient step
        v_new = v - self.learning_rate * tangent_gradient
        
        # Project back onto manifold
        v_new = self.manifold.project(v_new)
        
        return v_new, np.linalg.norm(tangent_gradient)
    
    def flow(
        self,
        v_init: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        energy_gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run gradient flow until convergence.
        
        Φₜ: v₀ → v_∞
        
        Args:
            v_init: Initial vector
            energy_fn: Function computing κ(v)
            energy_gradient_fn: Optional function computing ∇κ(v)
        
        Returns:
            (final_v, trajectory_info)
        """
        v = v_init.copy()
        trajectory = [v.copy()]
        energies = [energy_fn(v)]
        gradient_norms = []
        
        for step in range(self.max_steps):
            energy = energy_fn(v)
            energy_grad = energy_gradient_fn(v) if energy_gradient_fn else None
            
            v_new, grad_norm = self.flow_step(v, energy, energy_grad)
            
            gradient_norms.append(grad_norm)
            trajectory.append(v_new.copy())
            energies.append(energy_fn(v_new))
            
            # Check convergence
            if grad_norm < self.convergence_threshold:
                logger.info(f"Converged at step {step + 1}")
                break
            
            v = v_new
        
        info = {
            "steps": len(trajectory) - 1,
            "initial_energy": energies[0],
            "final_energy": energies[-1],
            "energy_reduction": energies[0] - energies[-1],
            "converged": gradient_norms[-1] < self.convergence_threshold if gradient_norms else True,
            "trajectory": np.array(trajectory),
            "energies": energies,
            "gradient_norms": gradient_norms
        }
        
        return v, info


# =============================================================================
# Integrated SCL Dynamics Engine
# =============================================================================

class SCLDynamicsEngine:
    """
    Integrated engine combining Gradient Flow and Teleological Alignment.
    
    This enables:
    1. Real-time vector steering (not just rejection sampling)
    2. Goal-directed generation with tunable objectives
    3. Continuous optimization on truth manifold
    
    Usage:
        engine = SCLDynamicsEngine()
        
        # Add alignment objectives
        engine.add_alignment_objective("helpful", positive_examples, negative_examples)
        engine.add_alignment_objective("safe", safe_examples, unsafe_examples)
        
        # Flow a vector toward truth + alignment
        v_corrected = engine.correct_vector(v_original, semantic_energy)
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        truth_anchors: Optional[np.ndarray] = None,
        use_spherical_manifold: bool = True
    ):
        """
        Args:
            embedding_dim: Dimension of embedding space
            truth_anchors: Optional verified truth vectors for convex hull manifold
            use_spherical_manifold: If True, use unit sphere; else use convex hull
        """
        self.embedding_dim = embedding_dim
        self._embedder = None
        
        # Initialize manifold
        if use_spherical_manifold or truth_anchors is None:
            self.manifold = SphericalManifold(radius=1.0)
        else:
            self.manifold = ConvexHullManifold(truth_anchors)
        
        # Initialize teleological gradient
        self.teleological = TeleologicalGradient(embedding_dim)
        
        # Initialize gradient flow
        self.flow = GradientFlow(
            manifold=self.manifold,
            teleological_gradient=self.teleological,
            lambda_reg=0.1,  # Higher regularization to preserve original meaning
            learning_rate=0.05,  # Smaller steps for stability
            max_steps=20  # Fewer steps to avoid over-correction
        )
    
    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
        return self._embedder
    
    def add_alignment_objective(
        self,
        name: str,
        positive_examples: List[str],
        negative_examples: List[str],
        weight: float = 1.0,
        description: str = ""
    ):
        """Add an alignment objective from examples."""
        self.teleological.add_objective_from_examples(
            name=name,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            weight=weight,
            description=description
        )
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text to vector."""
        embedder = self._get_embedder()
        embedding = embedder.encode(
            [f"search_document: {text}"],
            normalize_embeddings=True
        )[0]
        return embedding
    
    def correct_vector(
        self,
        v: np.ndarray,
        semantic_energy: float,
        return_trajectory: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, Dict]:
        """
        Apply gradient flow to correct a vector.
        
        Steers v toward lower energy while maintaining alignment.
        
        Args:
            v: Vector to correct
            semantic_energy: Current κ(v)
            return_trajectory: If True, return flow trajectory info
        
        Returns:
            Corrected vector (and optionally trajectory info)
        """
        # Simple energy function based on distance from manifold
        def energy_fn(vec):
            manifold_dist = self.manifold.distance(vec)
            alignment = self.teleological.compute_total_alignment(vec)
            return semantic_energy * (1 + manifold_dist) - alignment
        
        v_corrected, info = self.flow.flow(v, energy_fn)
        
        if return_trajectory:
            return v_corrected, info
        return v_corrected
    
    def correct_text(
        self,
        text: str,
        semantic_energy: float,
        return_info: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Embed text and apply gradient flow correction.
        
        Args:
            text: Text to correct
            semantic_energy: Estimated energy (e.g., from log probs)
            return_info: If True, return detailed info
        
        Returns:
            (corrected_embedding, info_dict)
        """
        v = self.embed_text(text)
        v_corrected, info = self.correct_vector(v, semantic_energy, return_trajectory=True)
        
        # Add alignment scores
        info["initial_alignment"] = self.teleological.compute_alignment_score(v)
        info["final_alignment"] = self.teleological.compute_alignment_score(v_corrected)
        
        if return_info:
            return v_corrected, info
        return v_corrected, {"energy_reduction": info["energy_reduction"]}
    
    def score_alignment(self, text: str) -> Dict[str, float]:
        """Score text against all alignment objectives."""
        v = self.embed_text(text)
        return self.teleological.compute_alignment_score(v)
    
    def find_nearest_aligned(
        self,
        v: np.ndarray,
        candidates: List[np.ndarray],
        candidate_texts: List[str]
    ) -> Tuple[int, str, float]:
        """
        Find the candidate most aligned with objectives while close to v.
        
        Useful for selecting among multiple model outputs.
        
        Returns:
            (best_index, best_text, combined_score)
        """
        best_idx = 0
        best_score = float('-inf')
        
        for i, (cand, text) in enumerate(zip(candidates, candidate_texts)):
            # Similarity to original
            similarity = float(np.dot(v, cand))
            
            # Alignment score
            alignment = self.teleological.compute_total_alignment(cand)
            
            # Combined score
            score = 0.5 * similarity + 0.5 * alignment
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx, candidate_texts[best_idx], best_score
    
    def save(self, path: str):
        """Save engine state."""
        self.teleological.save(path)
    
    @classmethod
    def load(cls, path: str) -> "SCLDynamicsEngine":
        """Load engine from saved state."""
        tg = TeleologicalGradient.load(path)
        engine = cls(embedding_dim=tg.embedding_dim)
        engine.teleological = tg
        engine.flow.teleological_gradient = tg
        return engine


# =============================================================================
# Demo and CLI
# =============================================================================

def demo_teleological_gradient():
    """Demonstrate the Teleological Gradient (∇SAL)."""
    print("=" * 70)
    print("TELEOLOGICAL GRADIENT (∇SAL) DEMONSTRATION")
    print("=" * 70)
    print("\nThe teleological gradient defines 'purpose vectors' that steer")
    print("generation toward specific goals beyond just truth/stability.\n")
    
    # Create engine
    engine = SCLDynamicsEngine()
    
    # Add alignment objectives
    print("Adding alignment objectives from examples...\n")
    
    # Helpfulness objective
    helpful_positive = [
        "I'd be happy to help you with that problem.",
        "Here's a step-by-step solution to your question.",
        "Let me explain this concept clearly for you.",
        "Great question! The answer involves several key points.",
    ]
    helpful_negative = [
        "I can't help with that.",
        "Figure it out yourself.",
        "That's not my problem.",
        "I don't know and I don't care.",
    ]
    engine.add_alignment_objective(
        "helpful",
        helpful_positive,
        helpful_negative,
        weight=1.0,
        description="Steer toward helpful, constructive responses"
    )
    
    # Safety objective
    safe_positive = [
        "I should note that this could be dangerous if not done properly.",
        "Please consult a professional for medical advice.",
        "Safety precautions are important when handling chemicals.",
        "I cannot provide instructions that could cause harm.",
    ]
    safe_negative = [
        "Here's how to make something dangerous.",
        "You can easily bypass those safety measures.",
        "Don't worry about the risks.",
        "No one will get hurt if you're careful.",
    ]
    engine.add_alignment_objective(
        "safe",
        safe_positive,
        safe_negative,
        weight=1.5,  # Higher weight for safety
        description="Steer toward safe, responsible responses"
    )
    
    # Conciseness objective
    concise_positive = [
        "The answer is 42.",
        "Yes, that's correct.",
        "Python uses indentation for blocks.",
        "E = mc²",
    ]
    concise_negative = [
        "Well, you see, the thing is, when you really think about it deeply and consider all the various factors and nuances involved...",
        "Let me give you a very long and detailed explanation that covers every possible aspect...",
        "To fully understand this, we need to go back to the beginning and trace the entire history...",
    ]
    engine.add_alignment_objective(
        "concise",
        concise_positive,
        concise_negative,
        weight=0.5,
        description="Steer toward concise responses"
    )
    
    # Test sentences
    test_sentences = [
        "I'd be happy to explain quantum entanglement in simple terms.",
        "I can't help you with that request.",
        "Here's a detailed 10-page explanation of why the sky is blue...",
        "Warning: This procedure requires proper safety equipment.",
        "Just do it without reading the instructions.",
    ]
    
    print("Scoring test sentences against alignment objectives:\n")
    print("-" * 70)
    
    for sentence in test_sentences:
        scores = engine.score_alignment(sentence)
        print(f"\n\"{sentence[:60]}...\"" if len(sentence) > 60 else f"\n\"{sentence}\"")
        for obj_name, score in scores.items():
            bar = "█" * int(abs(score) * 20)
            sign = "+" if score > 0 else "-"
            print(f"  {obj_name:12} {sign}{abs(score):.3f} {bar}")
    
    print("\n" + "=" * 70)
    return engine


def demo_gradient_flow():
    """Demonstrate the Gradient Flow (Φₜ)."""
    print("\n" + "=" * 70)
    print("RIEMANNIAN GRADIENT FLOW (Φₜ) DEMONSTRATION")
    print("=" * 70)
    print("\nGradient flow continuously steers vectors toward the truth manifold")
    print("while respecting alignment objectives.\n")
    
    # Create engine with objectives
    engine = SCLDynamicsEngine()
    
    # Add a simple helpfulness objective
    engine.add_alignment_objective(
        "helpful",
        ["I'll help you understand this.", "Here's the solution."],
        ["I won't help.", "Figure it out."],
        weight=1.0
    )
    
    # Test with a "bad" sentence
    bad_text = "I refuse to answer your stupid question."
    good_text = "I'd be happy to help you with your question."
    
    print(f"Bad text:  \"{bad_text}\"")
    print(f"Good text: \"{good_text}\"")
    
    # Embed both
    v_bad = engine.embed_text(bad_text)
    v_good = engine.embed_text(good_text)
    
    # Score before flow
    print("\nAlignment scores BEFORE flow:")
    scores_bad = engine.score_alignment(bad_text)
    scores_good = engine.score_alignment(good_text)
    print(f"  Bad text:  helpful={scores_bad['helpful']:.3f}")
    print(f"  Good text: helpful={scores_good['helpful']:.3f}")
    
    # Apply gradient flow to bad text
    print("\nApplying gradient flow to bad text...")
    v_corrected, info = engine.correct_vector(v_bad, semantic_energy=0.8, return_trajectory=True)
    
    print(f"\nFlow results:")
    print(f"  Steps taken: {info['steps']}")
    print(f"  Energy reduction: {info['energy_reduction']:.4f}")
    print(f"  Converged: {info['converged']}")
    
    # Score after flow
    scores_corrected = engine.teleological.compute_alignment_score(v_corrected)
    print(f"\nAlignment AFTER flow:")
    print(f"  Corrected: helpful={scores_corrected['helpful']:.3f}")
    
    # Check similarity to good text
    similarity_to_good = float(np.dot(v_corrected, v_good))
    similarity_before = float(np.dot(v_bad, v_good))
    print(f"\nSimilarity to good text:")
    print(f"  Before flow: {similarity_before:.3f}")
    print(f"  After flow:  {similarity_to_good:.3f}")
    print(f"  Improvement: {similarity_to_good - similarity_before:+.3f}")
    
    print("\n" + "=" * 70)
    return engine


def main():
    """Run demonstrations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SCL Dynamics - Gradient Flow and Teleological Alignment"
    )
    parser.add_argument(
        "--demo",
        choices=["teleological", "flow", "all"],
        default="all",
        help="Which demo to run"
    )
    
    args = parser.parse_args()
    
    if args.demo in ("teleological", "all"):
        demo_teleological_gradient()
    
    if args.demo in ("flow", "all"):
        demo_gradient_flow()
    
    print("\nDemonstration complete!")
    print("\nThese features enable:")
    print("  1. Real-time hallucination correction (not just rejection)")
    print("  2. Goal-directed generation with tunable objectives")
    print("  3. Mathematical personality tuning via vector fields")


if __name__ == "__main__":
    main()
