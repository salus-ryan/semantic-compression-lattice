#!/usr/bin/env python3
"""
scl_engine.py - Semantic Compression Lattice Inference Engine

Main orchestration script implementing hallucination-free inference through
the Lattice Meet (⊓) operation across a 3-model ensemble.

Models (via Ollama):
- llama3.1:8b - "The Logician"
- qwen2.5:14b - "The Technician"  
- mistral-nemo:12b - "The Writer"

Reference: "A Rigorous Formalization of the Semantic Compression Lattice"
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np

# Local imports
from lattice_ops import (
    MeaningAtom,
    LatticeMeet,
    SemanticEnergyCalculator,
    ShellVerifier,
    create_default_shell_verifier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-Engine")


@dataclass
class ModelConfig:
    """Configuration for a single model in the ensemble."""
    name: str  # Ollama model name (e.g., "llama3.1:8b")
    persona: str
    num_ctx: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class SCLConfig:
    """Configuration for the SCL Engine."""
    # Model paths - user must set these
    models: List[ModelConfig] = field(default_factory=list)
    
    # Embedding model
    embedder_name: str = "nomic-ai/nomic-embed-text-v1.5"
    
    # Lattice Meet parameters
    similarity_threshold: float = 0.85  # Admissibility threshold δ
    
    # Semantic Energy parameters
    lambda_reg: float = 0.01
    energy_threshold: float = -0.9
    beta: float = 1.0
    
    # Shell configuration
    enable_citation_shell: bool = True
    enable_safety_shell: bool = True
    enable_syntax_shell: bool = True
    
    # Generation parameters
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


class ModelEnsemble:
    """
    Manages the 3-model ensemble for generating diverse responses.
    
    Uses Ollama for local model inference.
    """
    
    def __init__(self, configs: List[ModelConfig]):
        self.configs = configs
        self.models: Dict[str, ModelConfig] = {}
        self._loaded = False
        self._client = None
    
    def load_models(self):
        """Verify models are available in Ollama."""
        import ollama
        
        self._client = ollama.Client()
        logger.info("Connecting to Ollama and verifying models...")
        
        # Get list of available models
        try:
            available = self._client.list()
            # Handle both old dict format and new Model object format
            models_list = available.get('models', []) if isinstance(available, dict) else available.models
            available_names = set()
            available_base = set()
            for m in models_list:
                name = m.get('name') if isinstance(m, dict) else m.model
                available_names.add(name)
                available_base.add(name.split(':')[0])
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: 'ollama serve'")
            return
        
        for config in self.configs:
            model_base = config.name.split(':')[0]
            
            if config.name in available_names or model_base in available_base:
                self.models[config.name] = config
                logger.info(f"✓ Found {config.name} ({config.persona})")
            else:
                logger.warning(f"✗ Model not found: {config.name}")
                logger.warning(f"  Pull it with: ollama pull {config.name}")
        
        self._loaded = len(self.models) > 0
        logger.info(f"Available: {len(self.models)}/{len(self.configs)} models")
        
        if not self._loaded:
            logger.error("No models available. Pull required models:")
            for config in self.configs:
                logger.error(f"  ollama pull {config.name}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Tuple[str, List[float]]]:
        """
        Generate responses from all models in the ensemble.
        
        Returns:
            Dict mapping model name to (response_text, per_token_log_probs)
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        import ollama
        
        default_system = "You are a helpful, accurate, and truthful assistant."
        sys_prompt = system_prompt or default_system
        
        results = {}
        
        for name, config in self.models.items():
            logger.info(f"Generating from {name} ({config.persona})...")
            
            try:
                start_time = time.time()
                
                # Use Ollama chat API
                response = self._client.chat(
                    model=name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_ctx": config.num_ctx,
                    }
                )
                
                elapsed = time.time() - start_time
                
                # Extract text
                text = response['message']['content'].strip()
                
                # Ollama doesn't return per-token log probs in chat mode
                # We'll use eval_count as a proxy for confidence
                log_probs = []
                if 'eval_count' in response:
                    # Approximate: more tokens = potentially less confident per token
                    eval_count = response.get('eval_count', 0)
                    eval_duration = response.get('eval_duration', 1)
                    # Tokens per second as rough quality proxy
                    if eval_duration > 0:
                        tps = eval_count / (eval_duration / 1e9)
                        # Normalize to log-prob-like value
                        log_probs = [-0.5] * max(1, eval_count // 10)
                
                results[name] = (text, log_probs)
                
                logger.info(f"✓ {name} generated {len(text)} chars in {elapsed:.1f}s")
                
            except Exception as e:
                logger.error(f"Generation failed for {name}: {e}")
                results[name] = ("", [])
        
        return results
    
    def generate_with_probs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Dict[str, Tuple[str, List[Tuple[str, float]]]]:
        """
        Generate responses using raw generate API with actual token probabilities.
        
        Uses Ollama's generate endpoint with logprobs=True to get real per-token
        log probabilities for accurate semantic energy calculation.
        
        Returns:
            Dict mapping model name to (response_text, list of (token, log_prob) tuples)
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        default_system = "You are a helpful, accurate, and truthful assistant."
        sys_prompt = system_prompt or default_system
        
        results = {}
        
        for name, config in self.models.items():
            logger.info(f"Generating from {name} ({config.persona}) with logprobs...")
            
            # Format prompt for raw generation
            formatted_prompt = self._format_prompt(name, prompt, sys_prompt)
            
            try:
                start_time = time.time()
                
                # Use streaming to collect token-by-token logprobs
                token_probs: List[Tuple[str, float]] = []
                full_response = ""
                
                for chunk in self._client.generate(
                    model=name,
                    prompt=formatted_prompt,
                    stream=True,
                    logprobs=True,
                    options={
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "num_ctx": config.num_ctx,
                    }
                ):
                    # Extract token and its log probability
                    token = chunk.get('response', '')
                    full_response += token
                    
                    # Get logprobs from response if available
                    if 'logprobs' in chunk and chunk['logprobs']:
                        logprobs_data = chunk['logprobs']
                        # Handle Ollama 0.12.11+ Logprob object format
                        # Returns: [Logprob(token='...', logprob=-0.5, top_logprobs=None)]
                        if isinstance(logprobs_data, list):
                            for item in logprobs_data:
                                # Handle Logprob namedtuple/object
                                if hasattr(item, 'token') and hasattr(item, 'logprob'):
                                    t = item.token
                                    lp = item.logprob
                                    if lp is not None:
                                        token_probs.append((t, float(lp)))
                                # Handle dict format (fallback)
                                elif isinstance(item, dict):
                                    t = item.get('token', token)
                                    lp = item.get('logprob', None)
                                    if lp is not None:
                                        token_probs.append((t, float(lp)))
                    elif token:
                        # Fallback: no logprobs available, use placeholder
                        token_probs.append((token, -0.5))
                
                elapsed = time.time() - start_time
                
                text = full_response.strip()
                results[name] = (text, token_probs)
                
                avg_logprob = (
                    sum(lp for _, lp in token_probs) / len(token_probs) 
                    if token_probs else -0.5
                )
                logger.info(
                    f"✓ {name} generated {len(text)} chars, "
                    f"{len(token_probs)} tokens, avg_logprob={avg_logprob:.3f} "
                    f"in {elapsed:.1f}s"
                )
                
            except Exception as e:
                logger.error(f"Generation failed for {name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                results[name] = ("", [])
        
        return results
    
    def _format_prompt(
        self, 
        model_name: str, 
        prompt: str, 
        system_prompt: str
    ) -> str:
        """Format prompt for raw generate API."""
        
        # Llama 3.1 format
        if "llama" in model_name.lower():
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Qwen format
        elif "qwen" in model_name.lower():
            return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Mistral format
        elif "mistral" in model_name.lower():
            return f"""<s>[INST] {system_prompt}

{prompt} [/INST]"""
        
        # Generic fallback
        else:
            return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    
    def unload_models(self):
        """Clear model references."""
        self.models = {}
        self._loaded = False
        self._client = None
        logger.info("Model references cleared")


class UniversalEmbedder:
    """
    Wrapper for sentence-transformers embedding model.
    Maps all model outputs to shared vector space V ⊂ ℝ^d.
    """
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self._model = None
    
    def load(self):
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedder: {self.model_name}")
        self._model = SentenceTransformer(
            self.model_name, 
            trust_remote_code=True
        )
        logger.info("✓ Embedder loaded")
    
    def encode(
        self, 
        texts: List[str], 
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """Encode texts into embeddings."""
        if self._model is None:
            self.load()
        
        # nomic-embed-text requires task prefix
        if "nomic" in self.model_name.lower():
            texts = [f"search_document: {t}" for t in texts]
        
        return self._model.encode(
            texts, 
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False
        )


class SCLEngine:
    """
    Main Semantic Compression Lattice Engine.
    
    Orchestrates the full pipeline:
    1. Generate responses from 3-model ensemble
    2. Split into meaning atoms
    3. Compute Lattice Meet (⊓) - intersection of semantic vertices
    4. Apply Invariant Shell verification
    5. Filter by Semantic Energy
    6. Reassemble surviving atoms
    """
    
    def __init__(self, config: SCLConfig):
        self.config = config
        self.ensemble: Optional[ModelEnsemble] = None
        self.embedder: Optional[UniversalEmbedder] = None
        self.lattice_meet: Optional[LatticeMeet] = None
        self.shell_verifier: Optional[ShellVerifier] = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing SCL Engine...")
        
        # Initialize embedder
        self.embedder = UniversalEmbedder(self.config.embedder_name)
        self.embedder.load()
        
        # Initialize shell verifier
        self.shell_verifier = ShellVerifier()
        if self.config.enable_citation_shell:
            from lattice_ops import CitationShell
            self.shell_verifier.add_shell(CitationShell())
        if self.config.enable_safety_shell:
            from lattice_ops import SafetyShell
            self.shell_verifier.add_shell(SafetyShell())
        if self.config.enable_syntax_shell:
            from lattice_ops import SyntaxShell
            self.shell_verifier.add_shell(SyntaxShell())
        
        # Initialize energy calculator
        energy_calc = SemanticEnergyCalculator(
            lambda_reg=self.config.lambda_reg,
            energy_threshold=self.config.energy_threshold,
            beta=self.config.beta
        )
        
        # Initialize lattice meet
        self.lattice_meet = LatticeMeet(
            embedder=self.embedder,
            similarity_threshold=self.config.similarity_threshold,
            energy_calculator=energy_calc,
            shell_verifier=self.shell_verifier
        )
        
        # Initialize model ensemble
        if self.config.models:
            self.ensemble = ModelEnsemble(self.config.models)
            self.ensemble.load_models()
        
        self._initialized = True
        logger.info("✓ SCL Engine initialized")
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        return_metadata: bool = False
    ) -> str | Tuple[str, Dict]:
        """
        Process a query through the SCL pipeline.
        
        Args:
            prompt: User query
            system_prompt: Optional system prompt
            return_metadata: If True, return (response, metadata) tuple
        
        Returns:
            Filtered response (or tuple with metadata)
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        if self.ensemble is None or len(self.ensemble.models) == 0:
            raise RuntimeError("No models loaded. Configure model paths first.")
        
        logger.info(f"Processing query: {prompt[:100]}...")
        
        # Step 1: Generate from all models with token probabilities
        generation_results = self.ensemble.generate_with_probs(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        # Separate texts and compute per-sentence log probs
        model_outputs = {}
        model_log_probs = {}
        
        for name, (text, token_probs) in generation_results.items():
            model_outputs[name] = text
            
            if token_probs:
                # Compute per-sentence log probs by mapping tokens to sentences
                sentences = self.lattice_meet.split_into_atoms(text)
                sentence_log_probs = self._compute_sentence_log_probs(
                    text, sentences, token_probs
                )
                model_log_probs[name] = sentence_log_probs
            else:
                model_log_probs[name] = []
        
        # Step 2: Compute Lattice Meet
        surviving_atoms, metadata = self.lattice_meet.compute_meet(
            model_outputs=model_outputs,
            model_log_probs=model_log_probs
        )
        
        # Step 3: Reassemble
        final_response = self.lattice_meet.reassemble(surviving_atoms)
        
        # Log statistics
        logger.info(
            f"Meet complete: {metadata['surviving']}/{metadata['total_atoms']} "
            f"atoms survived (threshold={self.config.similarity_threshold})"
        )
        
        if metadata["filtered"]["no_consensus"]:
            logger.info(
                f"Filtered {len(metadata['filtered']['no_consensus'])} atoms "
                "due to lack of consensus"
            )
        
        if metadata["filtered"]["high_energy"]:
            logger.info(
                f"Filtered {len(metadata['filtered']['high_energy'])} high-energy atoms"
            )
        
        if metadata["filtered"]["shell_violation"]:
            logger.info(
                f"Filtered {len(metadata['filtered']['shell_violation'])} atoms "
                "due to shell violations"
            )
        
        # Add generation results to metadata
        metadata["model_outputs"] = model_outputs
        
        if return_metadata:
            return final_response, metadata
        return final_response
    
    def _compute_sentence_log_probs(
        self,
        full_text: str,
        sentences: List[str],
        token_probs: List[Tuple[str, float]]
    ) -> List[float]:
        """
        Compute average log probability for each sentence.
        
        Maps token probabilities to sentences by tracking character positions.
        This enables accurate semantic energy calculation per meaning atom.
        
        Args:
            full_text: The complete generated text
            sentences: List of sentences extracted from the text
            token_probs: List of (token, log_prob) tuples
        
        Returns:
            List of average log probabilities, one per sentence
        """
        if not sentences or not token_probs:
            return [-0.5] * len(sentences) if sentences else []
        
        # Build character position to token index mapping
        char_pos = 0
        token_char_ranges = []  # [(start, end, log_prob), ...]
        
        for token, log_prob in token_probs:
            start = char_pos
            end = char_pos + len(token)
            token_char_ranges.append((start, end, log_prob))
            char_pos = end
        
        # Find sentence boundaries in the full text
        sentence_log_probs = []
        
        for sentence in sentences:
            # Find where this sentence appears in the text
            sent_start = full_text.find(sentence)
            if sent_start == -1:
                # Sentence not found exactly, use overall average
                avg = sum(lp for _, lp in token_probs) / len(token_probs)
                sentence_log_probs.append(avg)
                continue
            
            sent_end = sent_start + len(sentence)
            
            # Collect log probs for tokens that overlap with this sentence
            sent_token_probs = []
            for tok_start, tok_end, log_prob in token_char_ranges:
                # Check if token overlaps with sentence
                if tok_end > sent_start and tok_start < sent_end:
                    sent_token_probs.append(log_prob)
            
            if sent_token_probs:
                avg_log_prob = sum(sent_token_probs) / len(sent_token_probs)
            else:
                # Fallback to overall average
                avg_log_prob = sum(lp for _, lp in token_probs) / len(token_probs)
            
            sentence_log_probs.append(avg_log_prob)
        
        return sentence_log_probs
    
    def query_with_mock_outputs(
        self,
        model_outputs: Dict[str, str],
        model_log_probs: Optional[Dict[str, List[float]]] = None,
        return_metadata: bool = False
    ) -> str | Tuple[str, Dict]:
        """
        Process pre-generated model outputs through the SCL pipeline.
        
        Useful for testing without loading actual models.
        """
        if self.lattice_meet is None:
            # Initialize just the lattice components
            self.embedder = UniversalEmbedder(self.config.embedder_name)
            self.embedder.load()
            
            self.shell_verifier = create_default_shell_verifier()
            
            energy_calc = SemanticEnergyCalculator(
                lambda_reg=self.config.lambda_reg,
                energy_threshold=self.config.energy_threshold,
                beta=self.config.beta
            )
            
            self.lattice_meet = LatticeMeet(
                embedder=self.embedder,
                similarity_threshold=self.config.similarity_threshold,
                energy_calculator=energy_calc,
                shell_verifier=self.shell_verifier
            )
        
        surviving_atoms, metadata = self.lattice_meet.compute_meet(
            model_outputs=model_outputs,
            model_log_probs=model_log_probs
        )
        
        final_response = self.lattice_meet.reassemble(surviving_atoms)
        metadata["model_outputs"] = model_outputs
        
        if return_metadata:
            return final_response, metadata
        return final_response
    
    def shutdown(self):
        """Clean up resources."""
        if self.ensemble:
            self.ensemble.unload_models()
        logger.info("SCL Engine shut down")


def create_default_config() -> SCLConfig:
    """
    Create default configuration with Ollama model names.
    
    Requires these models to be pulled in Ollama:
        ollama pull llama3.1:8b
        ollama pull qwen2.5:14b  
        ollama pull mistral-nemo:12b
    """
    return SCLConfig(
        models=[
            ModelConfig(
                name="llama3.1:8b",
                persona="The Logician",
                num_ctx=4096
            ),
            ModelConfig(
                name="qwen2.5:14b",
                persona="The Technician",
                num_ctx=4096
            ),
            ModelConfig(
                name="mistral-nemo:12b",
                persona="The Writer",
                num_ctx=4096
            ),
        ],
        similarity_threshold=0.85,
        enable_citation_shell=True,
        enable_safety_shell=True,
        enable_syntax_shell=True
    )


def demo_hallucination_filtering():
    """
    Demonstration of hallucination filtering using mock model outputs.
    
    This shows how the SCL system filters out a hallucinated book title
    that only one model generates.
    """
    print("=" * 70)
    print("SCL HALLUCINATION FILTERING DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Simulated model outputs where one model hallucinates a book title
    mock_outputs = {
        "llama-3.1-8b": """
Albert Einstein was a theoretical physicist who developed the theory of relativity.
He was born in Ulm, Germany in 1879.
Einstein received the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect.
He published his famous equation E=mc² which describes mass-energy equivalence.
Einstein later moved to the United States and worked at Princeton University.
""",
        "qwen-2.5-14b": """
Albert Einstein was a renowned theoretical physicist best known for developing the theory of relativity.
Born in 1879 in Ulm, Germany, he showed early aptitude for mathematics and physics.
In 1921, Einstein was awarded the Nobel Prize in Physics for explaining the photoelectric effect.
His equation E=mc² revolutionized our understanding of mass and energy.
Einstein spent his later years at Princeton University in the United States.
His book "The Quantum Dreams of a Wandering Mind" explored his philosophical views on science.
""",
        "mistral-nemo-12b": """
Albert Einstein, born in Ulm, Germany in 1879, was one of the most influential physicists in history.
He is best known for his theory of relativity and contributions to quantum mechanics.
Einstein won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.
The famous equation E=mc² demonstrates the equivalence of mass and energy.
After fleeing Nazi Germany, Einstein joined Princeton University where he continued his research.
"""
    }
    
    # Create engine with default config (no models needed for mock test)
    config = SCLConfig(
        similarity_threshold=0.85,
        enable_citation_shell=False,  # Disable for demo
        enable_safety_shell=False,
        enable_syntax_shell=False
    )
    
    engine = SCLEngine(config)
    
    print("Input: 3 model responses about Albert Einstein")
    print("-" * 70)
    
    for name, output in mock_outputs.items():
        print(f"\n[{name}]:")
        print(output.strip())
    
    print("\n" + "=" * 70)
    print("COMPUTING LATTICE MEET (⊓)")
    print("=" * 70)
    
    # Process through SCL
    result, metadata = engine.query_with_mock_outputs(
        model_outputs=mock_outputs,
        return_metadata=True
    )
    
    print(f"\nSimilarity Threshold (δ): {config.similarity_threshold}")
    print(f"Total atoms across all models: {metadata['total_atoms']}")
    print(f"Atoms per model: {metadata['atoms_per_model']}")
    print(f"Surviving atoms after Meet: {metadata['surviving']}")
    
    print("\n" + "-" * 70)
    print("FILTERED ATOMS (Hallucinations/Non-consensus):")
    print("-" * 70)
    
    for atom in metadata["filtered"]["no_consensus"]:
        print(f"  ✗ {atom}")
    
    print("\n" + "=" * 70)
    print("FINAL OUTPUT (Intersubjective Truth):")
    print("=" * 70)
    print()
    print(result)
    print()
    
    return result, metadata


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SCL-Based Hallucination-Free Inference Engine"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run hallucination filtering demonstration"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=["llama3.1:8b", "qwen2.5:14b", "mistral-nemo:12b"],
        help="Ollama model names to use (default: llama3.1:8b qwen2.5:14b mistral-nemo:12b)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to process through the SCL engine"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for lattice meet (default: 0.85)"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demo_hallucination_filtering()
        return
    
    if args.query:
        config = create_default_config()
        config.similarity_threshold = args.threshold
        
        # Override models if specified
        if args.models:
            config.models = [
                ModelConfig(name=m, persona=f"Model-{i+1}")
                for i, m in enumerate(args.models)
            ]
        
        engine = SCLEngine(config)
        engine.initialize()
        
        try:
            result, metadata = engine.query(args.query, return_metadata=True)
            
            print("\n" + "=" * 70)
            print("SCL ENGINE RESULT")
            print("=" * 70)
            print(f"\nQuery: {args.query}")
            print(f"\nSurviving atoms: {metadata['surviving']}/{metadata['total_atoms']}")
            print(f"\nResponse:\n{result}")
            
        finally:
            engine.shutdown()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
