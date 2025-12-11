# Semantic Compression Lattice (SCL) Inference Engine

A hallucination-free AI inference system implementing the **Lattice Meet (⊓)** operation across a 3-model ensemble. By computing the mathematical intersection of "knowledge lattices," the system filters out hallucinations and retains only robust, intersubjective truths.

## Theoretical Foundation

Based on *"A Rigorous Formalization of the Semantic Compression Lattice"* (Barrett & Agents, 2025).

**Core Insight**: Hallucinations are rarely identical across different model families. By requiring semantic consensus across multiple models, we filter out fabrications while preserving verified facts.

### Key Concepts

| Concept | Definition | Implementation |
|---------|------------|----------------|
| **Meaning Atoms** | Individual sentences/claims (V ⊂ ℝ^d) | Sentence tokenization + embedding |
| **Lattice Meet (⊓)** | Intersection of vertex sets | Cosine similarity > 0.85 threshold |
| **Semantic Energy (κ)** | ‖∇L(v)‖₂ + λ‖v‖₂ | Token probability variance proxy |
| **Invariant Shells** | Hard constraints (V_S, φ_S, ε_S) | Citation/Safety/Syntax verification |

## Installation

### 1. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Pull Required Models

```bash
# The three-model ensemble
ollama pull llama3.1:8b      # The Logician (~4.7GB)
ollama pull qwen2.5:14b      # The Technician (~8.9GB)
ollama pull mistral-nemo:12b # The Writer (~7.1GB)
```

### 3. Install Python Dependencies

```bash
cd semantic-compression-lattice

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Model Ensemble

| Model | Ollama Name | Role | Size |
|-------|-------------|------|------|
| Llama 3.1 8B | `llama3.1:8b` | The Logician | ~4.7GB |
| Qwen 2.5 14B | `qwen2.5:14b` | The Technician | ~8.9GB |
| Mistral Nemo 12B | `mistral-nemo:12b` | The Writer | ~7.1GB |

**RAM Requirements**: ~20GB total (Ollama manages model loading efficiently).

## Usage

### Demo Mode (No Models Required)

Run the hallucination filtering demonstration with mock outputs:

```bash
python scl_engine.py --demo
```

### Full Inference

```bash
python scl_engine.py --query "Who was Albert Einstein?"
```

### Custom Models

```bash
# Use different Ollama models
python scl_engine.py --query "Explain quantum entanglement" \
    --models llama3.1:8b mistral:7b phi3:medium
```

### Python API

```python
from scl_engine import SCLEngine, create_default_config

# Create and initialize engine
config = create_default_config()
engine = SCLEngine(config)
engine.initialize()

# Query with metadata
response, metadata = engine.query(
    "What are the key principles of quantum mechanics?",
    return_metadata=True
)

print(f"Surviving atoms: {metadata['surviving']}/{metadata['total_atoms']}")
print(response)

engine.shutdown()
```

### Testing Without Models

```python
from scl_engine import SCLEngine, SCLConfig

config = SCLConfig(similarity_threshold=0.85)
engine = SCLEngine(config)

# Use pre-generated outputs
mock_outputs = {
    "model_a": "Einstein developed relativity. He won the Nobel Prize.",
    "model_b": "Einstein created the theory of relativity. Nobel Prize winner.",
    "model_c": "Einstein's relativity theory changed physics. He got a Nobel."
}

result, metadata = engine.query_with_mock_outputs(mock_outputs, return_metadata=True)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL ENSEMBLE (Phase A)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Llama-3.1   │  │ Qwen-2.5    │  │ Mistral     │              │
│  │ (Logician)  │  │ (Technician)│  │ (Writer)    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ATOMIZATION & EMBEDDING                          │
│  • Split responses into sentences (meaning atoms)                │
│  • Embed via nomic-embed-text-v1.5 → V ⊂ ℝ^768                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LATTICE MEET ⊓ (Phase B)                       │
│  For each atom v_A:                                              │
│    • Compute cosine similarity vs atoms in B and C               │
│    • Keep IFF sim(v_A, v_B) > 0.85 AND sim(v_A, v_C) > 0.85     │
│    • Discard non-consensus atoms (hallucinations)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 INVARIANT SHELLS (Phase C)                       │
│  Shell 1: Citation verification (HTTP 200 check)                 │
│  Shell 2: Safety classifier (sentiment alignment)                │
│  Shell 3: Syntax verification (AST parsing for code)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                SEMANTIC ENERGY FILTER (Phase D)                  │
│  • High energy (low log-prob) atoms require strict consensus     │
│  • κ(v) = ‖∇L(v)‖₂ + λ‖v‖₂ (approximated via token probs)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REASSEMBLY                                    │
│  Surviving atoms → Coherent response                             │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `scl_engine.py` | Main orchestration script |
| `lattice_ops.py` | Lattice Meet, Shells, Energy calculator |
| `demo_output.txt` | Example showing hallucination filtering |
| `requirements.txt` | Python dependencies |

## Configuration

```python
SCLConfig(
    # Lattice Meet
    similarity_threshold=0.85,  # Admissibility threshold δ
    
    # Semantic Energy
    lambda_reg=0.01,            # Position regularization λ
    energy_threshold=-0.9,      # High-energy cutoff
    beta=1.0,                   # Inverse temperature
    
    # Shells
    enable_citation_shell=True,
    enable_safety_shell=True,
    enable_syntax_shell=True,
    
    # Generation
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9
)
```

## Mathematical Reference

From the paper:

**Definition 2 (Curvature-Like Functional)**:
```
κ(v) = ‖∇L_world(v)‖₂ + λ‖v‖₂
```

**Theorem 1 (Completeness of SCL Lattice)**:
> Every pair of SCLs has a greatest lower bound (meet), defined by intersection of vertex sets, admissible edges, and shell intersections.

**Definition 6 (Invariant Shell)**:
> A triple S = (V_S, φ_S, ε_S) where V_S ⊆ V, φ_S: V_S → {0,1} is decidable, and ε_S ≥ 0 is the curvature tolerance.

## License

MIT
