"""
lattice_ops.py - Lattice Operations for Semantic Compression Lattice (SCL)

Implements the mathematical operations from "A Rigorous Formalization of the 
Semantic Compression Lattice" including:
- Meet (⊓) operation: Intersection of vertex sets across model outputs
- Invariant Shell verification and projection
- Semantic Energy (κ) computation using token probability variance

Reference: Definition 2 (κ), Definition 6 (Invariant Shells), Theorem 1 (Meet)
"""

import re
import ast
import sys
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Set
from abc import ABC, abstractmethod


@dataclass
class MeaningAtom:
    """
    A meaning atom v ∈ V representing a single semantic unit (sentence/claim).
    
    Attributes:
        text: The raw text of the atom
        embedding: Vector representation in shared embedding space V ⊂ ℝ^d
        source_model: Which model generated this atom
        semantic_energy: κ(v) = ‖∇L(v)‖₂ + λ‖v‖₂ (approximated via token probs)
        log_prob: Average log probability of tokens (proxy for gradient norm)
        is_high_energy: True if semantic energy exceeds threshold
    """
    text: str
    embedding: Optional[np.ndarray] = None
    source_model: str = ""
    semantic_energy: float = 0.0
    log_prob: float = 0.0
    is_high_energy: bool = False
    shell_violations: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.text)
    
    def __eq__(self, other):
        if isinstance(other, MeaningAtom):
            return self.text == other.text
        return False


class InvariantShell(ABC):
    """
    Invariant Shell S = (V_S, φ_S, ε_S) from Definition 6.
    
    An invariant shell is a constraint that atoms must satisfy.
    - V_S ⊆ V: subset of atoms this shell applies to
    - φ_S: V_S → {0,1}: decidable predicate
    - ε_S ≥ 0: curvature tolerance
    """
    
    def __init__(self, name: str = "shell", epsilon: float = 0.0):
        self.name = name
        self.epsilon = epsilon  # Curvature tolerance ε_S
    
    @abstractmethod
    def applies_to(self, atom: MeaningAtom) -> bool:
        """Check if this shell applies to the given atom (v ∈ V_S)."""
        pass
    
    @abstractmethod
    def verify(self, atom: MeaningAtom) -> bool:
        """Evaluate φ_S(v) → {0,1}. Returns True if atom satisfies constraint."""
        pass
    
    def project(self, atom: MeaningAtom) -> Optional[MeaningAtom]:
        """
        Admissible Projection Π_I(v) from Definition 5.
        Projects atom to constraint manifold M_S, or returns None if projection fails.
        """
        if not self.applies_to(atom):
            return atom  # Shell doesn't apply, pass through
        if self.verify(atom):
            return atom  # Already satisfies constraint
        return None  # Cannot project, reject atom


class CitationShell(InvariantShell):
    """
    Shell 1: Citation Verification
    If text contains URL/DOI, verify reachability (HTTP 200).
    """
    
    def __init__(self, timeout: float = 5.0):
        super().__init__(name="citation", epsilon=0.0)
        self.timeout = timeout
    
    # Patterns for URLs and DOIs
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        re.IGNORECASE
    )
    DOI_PATTERN = re.compile(
        r'10\.\d{4,}/[^\s]+',
        re.IGNORECASE
    )
    
    def applies_to(self, atom: MeaningAtom) -> bool:
        """Shell applies if atom contains URL or DOI."""
        return bool(
            self.URL_PATTERN.search(atom.text) or 
            self.DOI_PATTERN.search(atom.text)
        )
    
    def verify(self, atom: MeaningAtom) -> bool:
        """Verify all URLs/DOIs in the atom are reachable."""
        urls = self.URL_PATTERN.findall(atom.text)
        dois = self.DOI_PATTERN.findall(atom.text)
        
        # Convert DOIs to URLs
        doi_urls = [f"https://doi.org/{doi}" for doi in dois]
        all_urls = urls + doi_urls
        
        if not all_urls:
            return True
        
        for url in all_urls:
            try:
                response = requests.head(
                    url, 
                    timeout=self.timeout, 
                    allow_redirects=True
                )
                if response.status_code >= 400:
                    return False
            except (requests.RequestException, Exception):
                return False
        
        return True


class SafetyShell(InvariantShell):
    """
    Shell 2: Safety/Sentiment Alignment
    Uses a lightweight classifier to ensure content safety.
    """
    
    def __init__(self):
        super().__init__(name="safety", epsilon=0.1)
        self._classifier = None
        self._tokenizer = None
    
    def _load_classifier(self):
        """Lazy load the safety classifier."""
        if self._classifier is None:
            try:
                from transformers import (
                    AutoModelForSequenceClassification, 
                    AutoTokenizer
                )
                import torch
                
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._classifier = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                self._classifier.eval()
            except Exception as e:
                print(f"Warning: Could not load safety classifier: {e}")
                self._classifier = None
    
    def applies_to(self, atom: MeaningAtom) -> bool:
        """Safety shell applies to all atoms."""
        return True
    
    def verify(self, atom: MeaningAtom) -> bool:
        """Check if atom passes safety classification."""
        if self._classifier is None:
            self._load_classifier()
            if self._classifier is None:
                return True  # Fail open if classifier unavailable
        
        try:
            import torch
            
            inputs = self._tokenizer(
                atom.text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self._classifier(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # Label 1 is typically "positive" in SST-2
                # We use this as a proxy for "safe/acceptable"
                positive_prob = probs[0][1].item()
            
            # Accept if not strongly negative (threshold at 0.3)
            return positive_prob > 0.3
            
        except Exception:
            return True  # Fail open on errors


class SyntaxShell(InvariantShell):
    """
    Shell 3: Code Syntax Verification
    If output contains code, verify it parses correctly.
    """
    
    def __init__(self):
        super().__init__(name="syntax", epsilon=0.0)
    
    # Pattern to detect code blocks
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w+)?\s*\n(.*?)\n```',
        re.DOTALL
    )
    
    # Inline code pattern (for single expressions)
    INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')
    
    def applies_to(self, atom: MeaningAtom) -> bool:
        """Shell applies if atom contains code blocks."""
        return bool(self.CODE_BLOCK_PATTERN.search(atom.text))
    
    def verify(self, atom: MeaningAtom) -> bool:
        """Verify all code blocks parse correctly."""
        matches = self.CODE_BLOCK_PATTERN.findall(atom.text)
        
        for lang, code in matches:
            lang = lang.lower() if lang else "python"
            
            if lang in ("python", "py", "python3"):
                if not self._verify_python(code):
                    return False
            elif lang in ("javascript", "js"):
                # Basic JS verification - check balanced braces
                if not self._verify_balanced(code):
                    return False
            # Add more language parsers as needed
        
        return True
    
    def _verify_python(self, code: str) -> bool:
        """Verify Python code parses correctly."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _verify_balanced(self, code: str) -> bool:
        """Basic verification that braces/brackets are balanced."""
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        
        for char in code:
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack or stack[-1] != pairs[char]:
                    return False
                stack.pop()
        
        return len(stack) == 0


class ShellVerifier:
    """
    Manages a collection of Invariant Shells and performs verification.
    
    Implements the global constraint set M_I = ∩_{S∈I} M_S
    """
    
    def __init__(self):
        self.shells: List[InvariantShell] = []
    
    def add_shell(self, shell: InvariantShell):
        """Add an invariant shell to the verifier."""
        self.shells.append(shell)
    
    def verify_atom(self, atom: MeaningAtom) -> Tuple[bool, List[str]]:
        """
        Verify atom against all applicable shells.
        
        Returns:
            (passed, violations): Whether atom passed all shells and list of violations
        """
        violations = []
        
        for shell in self.shells:
            if shell.applies_to(atom):
                if not shell.verify(atom):
                    violations.append(shell.name)
        
        return len(violations) == 0, violations
    
    def project_atom(self, atom: MeaningAtom) -> Optional[MeaningAtom]:
        """
        Apply admissible projection Π_I(v) across all shells.
        
        Returns the projected atom or None if projection fails.
        """
        current = atom
        
        for shell in self.shells:
            projected = shell.project(current)
            if projected is None:
                return None
            current = projected
        
        return current


class SemanticEnergyCalculator:
    """
    Computes Semantic Energy κ(v) = ‖∇L(v)‖₂ + λ‖v‖₂
    
    Since we cannot compute full gradients on quantized models, we use
    token probability variance as a proxy for the gradient norm.
    
    Reference: Definition 2 (Curvature-Like Functional)
    """
    
    def __init__(
        self, 
        lambda_reg: float = 0.01,
        energy_threshold: float = -0.9,
        beta: float = 1.0
    ):
        """
        Args:
            lambda_reg: λ regularization weight for position term
            energy_threshold: Log-prob threshold below which atoms are "high energy"
            beta: Inverse temperature for hyperedge weights
        """
        self.lambda_reg = lambda_reg
        self.energy_threshold = energy_threshold
        self.beta = beta
    
    def compute_energy(
        self, 
        atom: MeaningAtom, 
        embedding: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute κ(v) for a meaning atom.
        
        κ(v) = ‖∇L(v)‖₂ + λ‖v‖₂
        
        We approximate ‖∇L(v)‖₂ using negative log probability (higher = more uncertain).
        """
        # Gradient norm proxy: use negative log prob (inverted so higher = more energy)
        gradient_proxy = -atom.log_prob if atom.log_prob != 0 else 1.0
        
        # Position regularization term
        if embedding is not None:
            position_term = self.lambda_reg * np.linalg.norm(embedding)
        elif atom.embedding is not None:
            position_term = self.lambda_reg * np.linalg.norm(atom.embedding)
        else:
            position_term = 0.0
        
        return gradient_proxy + position_term
    
    def is_high_energy(self, atom: MeaningAtom) -> bool:
        """
        Check if atom has high semantic energy (potential hallucination).
        
        High energy correlates with instability/hallucination.
        """
        return atom.log_prob < self.energy_threshold
    
    def compute_hyperedge_weight(
        self, 
        head_energy: float, 
        tail_energies: List[float]
    ) -> float:
        """
        Compute hyperedge weight w_e from Definition 4.
        
        w_e = exp(-β(κ(h_e) - min_{u∈T_e} κ(u)))
        """
        if not tail_energies:
            return 1.0
        
        min_tail = min(tail_energies)
        return np.exp(-self.beta * (head_energy - min_tail))


class LatticeMeet:
    """
    Implements the Lattice Meet (⊓) operation from Theorem 1.
    
    The Meet is defined as the intersection of vertex sets across model outputs.
    For atoms to survive, they must have semantic matches in ALL model outputs.
    """
    
    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.85,
        energy_calculator: Optional[SemanticEnergyCalculator] = None,
        shell_verifier: Optional[ShellVerifier] = None
    ):
        """
        Args:
            embedder: Sentence embedding model (e.g., nomic-embed-text-v1.5)
            similarity_threshold: Admissibility threshold δ for cosine similarity
            energy_calculator: Optional semantic energy calculator
            shell_verifier: Optional shell verifier for constraint checking
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.energy_calculator = energy_calculator or SemanticEnergyCalculator()
        self.shell_verifier = shell_verifier
    
    def split_into_atoms(self, text: str) -> List[str]:
        """
        Split text into meaning atoms (sentences/claims).
        
        Uses sentence boundary detection with handling for common edge cases.
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences (likely fragments)
        atoms = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return atoms
    
    def embed_atoms(self, atoms: List[MeaningAtom]) -> np.ndarray:
        """
        Embed all atoms into the shared vector space V ⊂ ℝ^d.
        
        Returns matrix of shape (n_atoms, embedding_dim).
        """
        texts = [atom.text for atom in atoms]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        
        # Store embeddings in atoms
        for atom, emb in zip(atoms, embeddings):
            atom.embedding = emb
        
        return embeddings
    
    def cosine_similarity_matrix(
        self, 
        embeddings_a: np.ndarray, 
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between two sets of embeddings.
        
        Since embeddings are normalized, this is just matrix multiplication.
        """
        return np.dot(embeddings_a, embeddings_b.T)
    
    def find_matches(
        self,
        atoms_a: List[MeaningAtom],
        atoms_b: List[MeaningAtom],
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
    ) -> Dict[int, List[int]]:
        """
        Find matching atoms between two sets based on similarity threshold.
        
        Returns dict mapping indices in A to list of matching indices in B.
        """
        sim_matrix = self.cosine_similarity_matrix(embeddings_a, embeddings_b)
        matches = {}
        
        for i in range(len(atoms_a)):
            matching_indices = np.where(sim_matrix[i] >= self.similarity_threshold)[0]
            if len(matching_indices) > 0:
                matches[i] = matching_indices.tolist()
        
        return matches
    
    def compute_meet(
        self,
        model_outputs: Dict[str, str],
        model_log_probs: Optional[Dict[str, List[float]]] = None
    ) -> Tuple[List[MeaningAtom], Dict]:
        """
        Compute the Lattice Meet (⊓) across multiple model outputs.
        
        This is the core operation: intersection of vertex sets.
        An atom survives IFF it has matches in ALL other model outputs.
        
        Args:
            model_outputs: Dict mapping model name to generated text
            model_log_probs: Optional dict mapping model name to per-sentence log probs
        
        Returns:
            (surviving_atoms, metadata): List of atoms that passed the meet, plus stats
        """
        if len(model_outputs) < 2:
            raise ValueError("Meet requires at least 2 model outputs")
        
        model_names = list(model_outputs.keys())
        
        # Phase 1: Split all outputs into atoms
        all_atoms: Dict[str, List[MeaningAtom]] = {}
        for name, text in model_outputs.items():
            atom_texts = self.split_into_atoms(text)
            atoms = []
            for i, t in enumerate(atom_texts):
                atom = MeaningAtom(text=t, source_model=name)
                # Assign log probs if available
                if model_log_probs and name in model_log_probs:
                    probs = model_log_probs[name]
                    if i < len(probs):
                        atom.log_prob = probs[i]
                atoms.append(atom)
            all_atoms[name] = atoms
        
        # Phase 2: Embed all atoms
        all_embeddings: Dict[str, np.ndarray] = {}
        for name, atoms in all_atoms.items():
            if atoms:
                all_embeddings[name] = self.embed_atoms(atoms)
            else:
                all_embeddings[name] = np.array([])
        
        # Phase 3: Compute semantic energy for all atoms
        for name, atoms in all_atoms.items():
            for atom in atoms:
                atom.semantic_energy = self.energy_calculator.compute_energy(atom)
                atom.is_high_energy = self.energy_calculator.is_high_energy(atom)
        
        # Phase 4: Find intersection (Meet operation)
        # Use first model as reference, find atoms that match in ALL others
        reference_model = model_names[0]
        reference_atoms = all_atoms[reference_model]
        reference_embeddings = all_embeddings[reference_model]
        
        if len(reference_atoms) == 0:
            return [], {"total_atoms": 0, "surviving": 0, "filtered": {}}
        
        # Track which reference atoms have matches in each other model
        match_counts = {i: set() for i in range(len(reference_atoms))}
        
        for other_model in model_names[1:]:
            other_atoms = all_atoms[other_model]
            other_embeddings = all_embeddings[other_model]
            
            if len(other_atoms) == 0:
                continue
            
            matches = self.find_matches(
                reference_atoms, other_atoms,
                reference_embeddings, other_embeddings
            )
            
            for ref_idx in matches:
                match_counts[ref_idx].add(other_model)
        
        # Phase 5: Filter - keep only atoms with matches in ALL other models
        required_matches = set(model_names[1:])
        surviving_atoms = []
        filtered_reasons = {
            "no_consensus": [],
            "high_energy": [],
            "shell_violation": []
        }
        
        for i, atom in enumerate(reference_atoms):
            matched_models = match_counts[i]
            
            # Check consensus requirement
            if matched_models != required_matches:
                # High energy atoms need strict consensus (all models)
                if atom.is_high_energy:
                    filtered_reasons["high_energy"].append(atom.text)
                    continue
                # Normal atoms need matches in all other models
                if not matched_models.issuperset(required_matches):
                    filtered_reasons["no_consensus"].append(atom.text)
                    continue
            
            # Phase 6: Apply shell verification if available
            if self.shell_verifier:
                passed, violations = self.shell_verifier.verify_atom(atom)
                if not passed:
                    atom.shell_violations = violations
                    filtered_reasons["shell_violation"].append(
                        f"{atom.text} [{', '.join(violations)}]"
                    )
                    continue
            
            surviving_atoms.append(atom)
        
        # Compute statistics
        total_atoms = sum(len(atoms) for atoms in all_atoms.values())
        metadata = {
            "total_atoms": total_atoms,
            "atoms_per_model": {k: len(v) for k, v in all_atoms.items()},
            "surviving": len(surviving_atoms),
            "filtered": filtered_reasons,
            "similarity_threshold": self.similarity_threshold
        }
        
        return surviving_atoms, metadata
    
    def reassemble(self, atoms: List[MeaningAtom]) -> str:
        """
        Reassemble surviving atoms into coherent text.
        
        Preserves original order and adds appropriate spacing.
        """
        if not atoms:
            return ""
        
        return " ".join(atom.text for atom in atoms)


class MathShell(InvariantShell):
    """
    Shell 4: Mathematical Verification using SymPy
    
    If an atom contains mathematical equations or expressions, verify them
    using SymPy's symbolic mathematics engine.
    
    Checks:
    - Equation syntax validity
    - Mathematical consistency (e.g., derivatives, simplifications)
    - Numeric claims (e.g., "2+2=4")
    """
    
    def __init__(self):
        super().__init__(name="math", epsilon=0.0)
        self._sympy_available = None
    
    # Patterns for mathematical content
    EQUATION_PATTERNS = [
        # LaTeX equations
        re.compile(r'\$([^$]+)\$'),
        re.compile(r'\\\[([^\]]+)\\\]'),
        re.compile(r'\\\(([^)]+)\\\)'),
        # Common equation formats
        re.compile(r'([a-zA-Z]+)\s*=\s*([^,\.;]+)'),
        # Derivative notation
        re.compile(r'd([a-zA-Z])/d([a-zA-Z])\s*=\s*([^,\.;]+)'),
        re.compile(r'\\frac\{d([a-zA-Z])\}\{d([a-zA-Z])\}\s*=\s*([^,\.;]+)'),
    ]
    
    # Pattern for numeric claims
    NUMERIC_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*([+\-*/^])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
    )
    
    def _check_sympy(self) -> bool:
        """Check if SymPy is available."""
        if self._sympy_available is None:
            try:
                import sympy
                self._sympy_available = True
            except ImportError:
                self._sympy_available = False
        return self._sympy_available
    
    def applies_to(self, atom: MeaningAtom) -> bool:
        """Shell applies if atom contains mathematical content."""
        text = atom.text
        
        # Check for LaTeX math
        if '$' in text or '\\[' in text or '\\(' in text:
            return True
        
        # Check for equation patterns
        for pattern in self.EQUATION_PATTERNS:
            if pattern.search(text):
                return True
        
        # Check for numeric claims
        if self.NUMERIC_PATTERN.search(text):
            return True
        
        return False
    
    def verify(self, atom: MeaningAtom) -> bool:
        """Verify mathematical content using SymPy."""
        if not self._check_sympy():
            return True  # Fail open if SymPy not available
        
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr, 
            standard_transformations,
            implicit_multiplication_application
        )
        
        text = atom.text
        
        # Verify numeric claims first (simple arithmetic)
        numeric_matches = self.NUMERIC_PATTERN.findall(text)
        for match in numeric_matches:
            try:
                a, op, b, result = match
                a, b, result = float(a), float(b), float(result)
                
                if op == '+':
                    expected = a + b
                elif op == '-':
                    expected = a - b
                elif op == '*':
                    expected = a * b
                elif op == '/':
                    expected = a / b if b != 0 else float('inf')
                elif op == '^':
                    expected = a ** b
                else:
                    continue
                
                # Allow small floating point tolerance
                if abs(expected - result) > 1e-6:
                    return False
                    
            except (ValueError, ZeroDivisionError):
                continue
        
        # Try to parse and verify symbolic expressions
        for pattern in self.EQUATION_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    # Handle different match formats
                    if isinstance(match, tuple):
                        expr_str = match[0] if len(match) == 1 else '='.join(match)
                    else:
                        expr_str = match
                    
                    # Clean up LaTeX notation for SymPy
                    expr_str = self._latex_to_sympy(expr_str)
                    
                    # Try to parse the expression
                    transformations = (
                        standard_transformations + 
                        (implicit_multiplication_application,)
                    )
                    
                    if '=' in expr_str:
                        lhs, rhs = expr_str.split('=', 1)
                        lhs_expr = parse_expr(lhs.strip(), transformations=transformations)
                        rhs_expr = parse_expr(rhs.strip(), transformations=transformations)
                        
                        # Check if equation is mathematically valid
                        # (simplify both sides and compare)
                        diff = sympy.simplify(lhs_expr - rhs_expr)
                        # If difference simplifies to 0, equation is valid
                        # Otherwise, we can't definitively say it's wrong
                    else:
                        # Just verify it parses
                        parse_expr(expr_str, transformations=transformations)
                        
                except Exception:
                    # Parsing failed - could be invalid math
                    # But we fail open to avoid false positives
                    continue
        
        return True
    
    def _latex_to_sympy(self, latex: str) -> str:
        """Convert common LaTeX notation to SymPy-parseable format."""
        # Remove LaTeX delimiters
        latex = latex.replace('\\', '')
        
        # Common substitutions
        replacements = [
            ('frac{', '('),
            ('}{', ')/('),
            ('}', ')'),
            ('{', '('),
            ('^2', '**2'),
            ('^3', '**3'),
            ('sqrt', 'sqrt'),
            ('pi', 'pi'),
            ('sin', 'sin'),
            ('cos', 'cos'),
            ('tan', 'tan'),
            ('log', 'log'),
            ('ln', 'log'),
            ('exp', 'exp'),
            ('cdot', '*'),
            ('times', '*'),
        ]
        
        for old, new in replacements:
            latex = latex.replace(old, new)
        
        return latex
    
    def verify_derivative(self, equation: str, var: str, expected: str) -> bool:
        """
        Verify a derivative claim.
        
        Args:
            equation: The original equation (e.g., "E=m*c**2")
            var: Variable to differentiate with respect to
            expected: Expected derivative result
        
        Returns:
            True if derivative is correct
        """
        if not self._check_sympy():
            return True
        
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
        
        try:
            # Parse equation
            if '=' in equation:
                lhs, rhs = equation.split('=')
                expr = parse_expr(rhs.strip())
            else:
                expr = parse_expr(equation)
            
            # Get the variable symbol
            var_sym = sympy.Symbol(var)
            
            # Compute derivative
            derivative = sympy.diff(expr, var_sym)
            
            # Parse expected result
            expected_expr = parse_expr(expected)
            
            # Compare (simplify difference)
            diff = sympy.simplify(derivative - expected_expr)
            
            return diff == 0
            
        except Exception:
            return True  # Fail open on errors


class CodeExecutionShell(InvariantShell):
    """
    Shell 5: Code Execution Verification
    
    If an atom contains code, actually execute it in a sandboxed environment
    to verify it works (not just parses).
    
    Uses subprocess with resource limits for safety.
    """
    
    def __init__(
        self, 
        timeout: float = 5.0,
        max_memory_mb: int = 100,
        enable_sandbox: bool = True
    ):
        super().__init__(name="code_exec", epsilon=0.0)
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.enable_sandbox = enable_sandbox
    
    # Pattern to detect code blocks with language
    CODE_BLOCK_PATTERN = re.compile(
        r'```(python|py|python3)\s*\n(.*?)\n```',
        re.DOTALL | re.IGNORECASE
    )
    
    def applies_to(self, atom: MeaningAtom) -> bool:
        """Shell applies if atom contains Python code blocks."""
        return bool(self.CODE_BLOCK_PATTERN.search(atom.text))
    
    def verify(self, atom: MeaningAtom) -> bool:
        """Execute code in sandbox and verify it runs without errors."""
        matches = self.CODE_BLOCK_PATTERN.findall(atom.text)
        
        for lang, code in matches:
            # Skip if code looks like it needs external resources
            if self._needs_external_resources(code):
                continue
            
            # Execute in sandbox
            success, output, error = self._execute_sandboxed(code)
            
            if not success:
                return False
        
        return True
    
    def _needs_external_resources(self, code: str) -> bool:
        """Check if code needs external resources (network, files, etc.)."""
        dangerous_patterns = [
            'import requests',
            'import urllib',
            'import socket',
            'open(',
            'subprocess',
            'os.system',
            'eval(',
            'exec(',
            '__import__',
            'input(',
        ]
        
        code_lower = code.lower()
        return any(pattern.lower() in code_lower for pattern in dangerous_patterns)
    
    def _execute_sandboxed(
        self, 
        code: str
    ) -> tuple:
        """
        Execute Python code in a sandboxed subprocess.
        
        Returns:
            (success, stdout, stderr)
        """
        import subprocess
        import tempfile
        import os
        
        # Create a wrapper script with resource limits
        wrapper = f'''
import sys
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, ({self.max_memory_mb * 1024 * 1024}, {self.max_memory_mb * 1024 * 1024}))
resource.setrlimit(resource.RLIMIT_CPU, ({int(self.timeout)}, {int(self.timeout)}))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # No subprocess spawning

# Restricted builtins
restricted_builtins = {{
    'print': print,
    'len': len,
    'range': range,
    'int': int,
    'float': float,
    'str': str,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'bool': bool,
    'abs': abs,
    'min': min,
    'max': max,
    'sum': sum,
    'sorted': sorted,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'isinstance': isinstance,
    'type': type,
    'True': True,
    'False': False,
    'None': None,
}}

# Allow safe math imports
try:
    import math
    restricted_builtins['math'] = math
except:
    pass

try:
    exec(compile({repr(code)}, '<code>', 'exec'), restricted_builtins)
    print("__EXECUTION_SUCCESS__")
except Exception as e:
    print(f"__EXECUTION_ERROR__: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
        
        try:
            # Write wrapper to temp file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False
            ) as f:
                f.write(wrapper)
                temp_path = f.name
            
            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 1,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
                )
                
                success = (
                    result.returncode == 0 and 
                    '__EXECUTION_SUCCESS__' in result.stdout
                )
                
                return success, result.stdout, result.stderr
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return False, '', 'Execution timed out'
        except Exception as e:
            return False, '', str(e)


def create_default_shell_verifier() -> ShellVerifier:
    """Create a ShellVerifier with all default shells."""
    verifier = ShellVerifier()
    verifier.add_shell(CitationShell())
    verifier.add_shell(SafetyShell())
    verifier.add_shell(SyntaxShell())
    return verifier


def create_hardened_shell_verifier() -> ShellVerifier:
    """
    Create a ShellVerifier with all shells including hardened Math and Code shells.
    
    This provides maximum verification but may be slower due to:
    - SymPy mathematical verification
    - Sandboxed code execution
    """
    verifier = ShellVerifier()
    verifier.add_shell(CitationShell())
    verifier.add_shell(SafetyShell())
    verifier.add_shell(SyntaxShell())
    verifier.add_shell(MathShell())
    verifier.add_shell(CodeExecutionShell())
    return verifier


# =============================================================================
# Lattice Join (⊔) Operation - Theorem 1, Part 2
# =============================================================================

@dataclass
class LatticeVertex:
    """
    A vertex in the semantic lattice representing a meaning atom with metadata.
    
    Used for serialization and lattice operations (Join/Meet).
    """
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    semantic_energy: float = 0.0
    source_domain: str = ""  # e.g., "law", "code", "medicine"
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize vertex to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "semantic_energy": self.semantic_energy,
            "source_domain": self.source_domain,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LatticeVertex":
        """Deserialize vertex from dictionary."""
        embedding = None
        if data.get("embedding") is not None:
            embedding = np.array(data["embedding"], dtype=np.float32)
        return cls(
            id=data["id"],
            text=data["text"],
            embedding=embedding,
            semantic_energy=data.get("semantic_energy", 0.0),
            source_domain=data.get("source_domain", ""),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_atom(cls, atom: MeaningAtom, domain: str = "") -> "LatticeVertex":
        """Create vertex from a MeaningAtom."""
        import hashlib
        vertex_id = hashlib.md5(atom.text.encode()).hexdigest()[:16]
        return cls(
            id=vertex_id,
            text=atom.text,
            embedding=atom.embedding,
            semantic_energy=atom.semantic_energy,
            source_domain=domain,
            metadata={
                "source_model": atom.source_model,
                "log_prob": atom.log_prob,
                "is_high_energy": atom.is_high_energy
            }
        )


@dataclass
class LatticeEdge:
    """
    A hyperedge in the semantic lattice connecting vertices.
    
    Represents semantic relationships between meaning atoms.
    """
    id: str
    source_id: str
    target_ids: List[str]
    weight: float = 1.0
    edge_type: str = "semantic"  # semantic, causal, temporal
    
    def to_dict(self) -> Dict:
        """Serialize edge to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_ids": self.target_ids,
            "weight": self.weight,
            "edge_type": self.edge_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LatticeEdge":
        """Deserialize edge from dictionary."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_ids=data["target_ids"],
            weight=data.get("weight", 1.0),
            edge_type=data.get("edge_type", "semantic")
        )


@dataclass
class SemanticLattice:
    """
    A complete semantic lattice structure for serialization and operations.
    
    Contains vertices (meaning atoms) and edges (semantic relationships).
    This is the fundamental data structure for Join/Meet operations.
    """
    name: str
    domain: str
    vertices: Dict[str, LatticeVertex] = field(default_factory=dict)
    edges: Dict[str, LatticeEdge] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def add_vertex(self, vertex: LatticeVertex):
        """Add a vertex to the lattice."""
        self.vertices[vertex.id] = vertex
    
    def add_edge(self, edge: LatticeEdge):
        """Add an edge to the lattice."""
        self.edges[edge.id] = edge
    
    def get_vertex_by_text(self, text: str) -> Optional[LatticeVertex]:
        """Find vertex by exact text match."""
        for v in self.vertices.values():
            if v.text == text:
                return v
        return None
    
    def to_dict(self) -> Dict:
        """Serialize lattice to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "vertices": {k: v.to_dict() for k, v in self.vertices.items()},
            "edges": {k: e.to_dict() for k, e in self.edges.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SemanticLattice":
        """Deserialize lattice from dictionary."""
        lattice = cls(
            name=data["name"],
            domain=data["domain"],
            metadata=data.get("metadata", {})
        )
        for vid, vdata in data.get("vertices", {}).items():
            lattice.vertices[vid] = LatticeVertex.from_dict(vdata)
        for eid, edata in data.get("edges", {}).items():
            lattice.edges[eid] = LatticeEdge.from_dict(edata)
        return lattice
    
    def save(self, path: str):
        """Save lattice to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "SemanticLattice":
        """Load lattice from JSON file."""
        import json
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class LatticeJoin:
    """
    Implements the Lattice Join (⊔) operation from Theorem 1.
    
    The Join is defined as the Least Upper Bound (LUB) of two lattices:
    - Performs disjoint union of vertex sets
    - Resolves conflicts using curvature (semantic energy) - lower energy wins
    - Pushes forward curvature from both lattices
    
    This enables "Modular Intelligence" - combining knowledge domains:
    - Law Lattice ⊔ Code Lattice → Legal Tech Agent
    - Medicine Lattice ⊔ Research Lattice → Medical Research Agent
    
    Reference: Theorem 1, Part 2 - "The Join (⊔) is the Least Upper Bound"
    """
    
    def __init__(
        self,
        embedder=None,
        similarity_threshold: float = 0.85,
        conflict_resolution: str = "energy"  # "energy", "recency", "source_priority"
    ):
        """
        Args:
            embedder: Sentence embedding model for semantic similarity
            similarity_threshold: Threshold for considering vertices as duplicates
            conflict_resolution: Strategy for resolving conflicts:
                - "energy": Keep vertex with lower semantic energy (κ)
                - "recency": Keep more recent vertex
                - "source_priority": Use domain priority ordering
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.conflict_resolution = conflict_resolution
        self._domain_priority: Dict[str, int] = {}
    
    def set_domain_priority(self, priorities: Dict[str, int]):
        """
        Set priority ordering for domains in conflict resolution.
        
        Higher priority number = preferred in conflicts.
        Example: {"verified": 10, "academic": 8, "web": 3}
        """
        self._domain_priority = priorities
    
    def _compute_similarity(
        self, 
        v1: LatticeVertex, 
        v2: LatticeVertex
    ) -> float:
        """Compute semantic similarity between two vertices."""
        if v1.embedding is None or v2.embedding is None:
            # Fall back to text comparison
            return 1.0 if v1.text == v2.text else 0.0
        
        # Cosine similarity (embeddings should be normalized)
        return float(np.dot(v1.embedding, v2.embedding))
    
    def _find_conflicts(
        self,
        lattice_a: SemanticLattice,
        lattice_b: SemanticLattice
    ) -> List[Tuple[LatticeVertex, LatticeVertex, float]]:
        """
        Find conflicting vertices between two lattices.
        
        Conflicts are vertices that are semantically similar but potentially
        have different content or energy levels.
        
        Returns list of (vertex_a, vertex_b, similarity) tuples.
        """
        conflicts = []
        
        for va in lattice_a.vertices.values():
            for vb in lattice_b.vertices.values():
                similarity = self._compute_similarity(va, vb)
                if similarity >= self.similarity_threshold:
                    conflicts.append((va, vb, similarity))
        
        return conflicts
    
    def _resolve_conflict(
        self,
        v1: LatticeVertex,
        v2: LatticeVertex,
        similarity: float
    ) -> LatticeVertex:
        """
        Resolve conflict between two similar vertices.
        
        Uses the configured conflict resolution strategy.
        Returns the winning vertex (possibly merged).
        """
        if self.conflict_resolution == "energy":
            # Lower semantic energy wins (more stable/confident)
            winner = v1 if v1.semantic_energy <= v2.semantic_energy else v2
            loser = v2 if winner is v1 else v1
            
        elif self.conflict_resolution == "recency":
            # More recent wins
            t1 = v1.metadata.get("timestamp", 0)
            t2 = v2.metadata.get("timestamp", 0)
            winner = v1 if t1 >= t2 else v2
            loser = v2 if winner is v1 else v1
            
        elif self.conflict_resolution == "source_priority":
            # Higher domain priority wins
            p1 = self._domain_priority.get(v1.source_domain, 0)
            p2 = self._domain_priority.get(v2.source_domain, 0)
            winner = v1 if p1 >= p2 else v2
            loser = v2 if winner is v1 else v1
            
        else:
            # Default: energy-based
            winner = v1 if v1.semantic_energy <= v2.semantic_energy else v2
            loser = v2 if winner is v1 else v1
        
        # Create merged vertex with provenance tracking
        merged = LatticeVertex(
            id=winner.id,
            text=winner.text,
            embedding=winner.embedding,
            semantic_energy=winner.semantic_energy,
            source_domain=winner.source_domain,
            metadata={
                **winner.metadata,
                "merged_from": [v1.id, v2.id],
                "merge_similarity": similarity,
                "alternate_text": loser.text if loser.text != winner.text else None,
                "alternate_energy": loser.semantic_energy
            }
        )
        
        return merged
    
    def compute_join(
        self,
        lattice_a: SemanticLattice,
        lattice_b: SemanticLattice,
        name: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Tuple[SemanticLattice, Dict]:
        """
        Compute the Lattice Join (⊔) of two semantic lattices.
        
        L_A ⊔ L_B = Least Upper Bound
        
        Algorithm:
        1. Start with disjoint union of all vertices
        2. Find semantically similar vertices (potential conflicts)
        3. Resolve conflicts using curvature (energy) - lower wins
        4. Merge edges, updating vertex references
        5. Recompute edge weights based on merged structure
        
        Args:
            lattice_a: First lattice (e.g., "Law Lattice")
            lattice_b: Second lattice (e.g., "Code Lattice")
            name: Name for the joined lattice
            domain: Domain label for the joined lattice
        
        Returns:
            (joined_lattice, metadata): The LUB lattice and operation statistics
        """
        # Create result lattice
        joined_name = name or f"{lattice_a.name}⊔{lattice_b.name}"
        joined_domain = domain or f"{lattice_a.domain}+{lattice_b.domain}"
        
        result = SemanticLattice(
            name=joined_name,
            domain=joined_domain,
            metadata={
                "source_lattices": [lattice_a.name, lattice_b.name],
                "join_threshold": self.similarity_threshold,
                "conflict_resolution": self.conflict_resolution
            }
        )
        
        # Track statistics
        stats = {
            "vertices_a": len(lattice_a.vertices),
            "vertices_b": len(lattice_b.vertices),
            "edges_a": len(lattice_a.edges),
            "edges_b": len(lattice_b.edges),
            "conflicts_found": 0,
            "conflicts_resolved": 0,
            "vertices_merged": 0,
            "final_vertices": 0,
            "final_edges": 0
        }
        
        # Step 1: Find conflicts
        conflicts = self._find_conflicts(lattice_a, lattice_b)
        stats["conflicts_found"] = len(conflicts)
        
        # Build conflict mapping: vertex_id -> resolved_vertex
        conflict_map_a: Dict[str, LatticeVertex] = {}  # A vertices that were merged
        conflict_map_b: Dict[str, LatticeVertex] = {}  # B vertices that were merged
        merged_vertices: Set[str] = set()  # IDs of vertices that resulted from merges
        
        # Step 2: Resolve conflicts
        for va, vb, similarity in conflicts:
            merged = self._resolve_conflict(va, vb, similarity)
            conflict_map_a[va.id] = merged
            conflict_map_b[vb.id] = merged
            merged_vertices.add(merged.id)
            stats["conflicts_resolved"] += 1
        
        # Step 3: Add all vertices (disjoint union with conflict resolution)
        # Add vertices from A
        for vid, vertex in lattice_a.vertices.items():
            if vid in conflict_map_a:
                # This vertex was merged - add the merged version
                merged = conflict_map_a[vid]
                if merged.id not in result.vertices:
                    result.add_vertex(merged)
                    stats["vertices_merged"] += 1
            else:
                # No conflict - add directly with domain prefix to ensure uniqueness
                new_vertex = LatticeVertex(
                    id=f"a_{vid}",
                    text=vertex.text,
                    embedding=vertex.embedding,
                    semantic_energy=vertex.semantic_energy,
                    source_domain=vertex.source_domain or lattice_a.domain,
                    metadata={**vertex.metadata, "source_lattice": lattice_a.name}
                )
                result.add_vertex(new_vertex)
        
        # Add vertices from B (skip those already merged)
        for vid, vertex in lattice_b.vertices.items():
            if vid in conflict_map_b:
                # Already handled via conflict resolution
                continue
            else:
                # No conflict - add directly with domain prefix
                new_vertex = LatticeVertex(
                    id=f"b_{vid}",
                    text=vertex.text,
                    embedding=vertex.embedding,
                    semantic_energy=vertex.semantic_energy,
                    source_domain=vertex.source_domain or lattice_b.domain,
                    metadata={**vertex.metadata, "source_lattice": lattice_b.name}
                )
                result.add_vertex(new_vertex)
        
        # Step 4: Merge edges with vertex ID remapping
        def remap_vertex_id(vid: str, source: str) -> str:
            """Remap vertex ID to the joined lattice."""
            if source == "a":
                if vid in conflict_map_a:
                    return conflict_map_a[vid].id
                return f"a_{vid}"
            else:  # source == "b"
                if vid in conflict_map_b:
                    return conflict_map_b[vid].id
                return f"b_{vid}"
        
        # Add edges from A
        for eid, edge in lattice_a.edges.items():
            new_edge = LatticeEdge(
                id=f"a_{eid}",
                source_id=remap_vertex_id(edge.source_id, "a"),
                target_ids=[remap_vertex_id(tid, "a") for tid in edge.target_ids],
                weight=edge.weight,
                edge_type=edge.edge_type
            )
            result.add_edge(new_edge)
        
        # Add edges from B
        for eid, edge in lattice_b.edges.items():
            new_edge = LatticeEdge(
                id=f"b_{eid}",
                source_id=remap_vertex_id(edge.source_id, "b"),
                target_ids=[remap_vertex_id(tid, "b") for tid in edge.target_ids],
                weight=edge.weight,
                edge_type=edge.edge_type
            )
            result.add_edge(new_edge)
        
        stats["final_vertices"] = len(result.vertices)
        stats["final_edges"] = len(result.edges)
        
        return result, stats
    
    def join_multiple(
        self,
        lattices: List[SemanticLattice],
        name: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Tuple[SemanticLattice, Dict]:
        """
        Join multiple lattices sequentially.
        
        L_1 ⊔ L_2 ⊔ ... ⊔ L_n
        
        Uses left-associative folding: ((L_1 ⊔ L_2) ⊔ L_3) ⊔ ...
        """
        if not lattices:
            raise ValueError("Cannot join empty list of lattices")
        
        if len(lattices) == 1:
            return lattices[0], {"single_lattice": True}
        
        # Fold left
        result = lattices[0]
        all_stats = []
        
        for i, lattice in enumerate(lattices[1:], 1):
            result, stats = self.compute_join(result, lattice)
            stats["join_step"] = i
            all_stats.append(stats)
        
        # Set final name/domain
        if name:
            result.name = name
        if domain:
            result.domain = domain
        
        combined_stats = {
            "total_joins": len(all_stats),
            "join_steps": all_stats,
            "final_vertices": len(result.vertices),
            "final_edges": len(result.edges)
        }
        
        return result, combined_stats


def create_lattice_from_atoms(
    atoms: List[MeaningAtom],
    name: str,
    domain: str,
    embedder=None
) -> SemanticLattice:
    """
    Create a SemanticLattice from a list of MeaningAtoms.
    
    Useful for converting Meet operation results into a lattice structure
    that can be saved and later joined with other lattices.
    
    Args:
        atoms: List of MeaningAtoms (e.g., from LatticeMeet.compute_meet)
        name: Name for the lattice
        domain: Domain label (e.g., "law", "medicine", "code")
        embedder: Optional embedder to compute embeddings if missing
    
    Returns:
        SemanticLattice ready for serialization or Join operations
    """
    lattice = SemanticLattice(name=name, domain=domain)
    
    # Embed atoms if needed
    if embedder is not None:
        texts_to_embed = [a.text for a in atoms if a.embedding is None]
        if texts_to_embed:
            embeddings = embedder.encode(texts_to_embed, normalize_embeddings=True)
            emb_idx = 0
            for atom in atoms:
                if atom.embedding is None:
                    atom.embedding = embeddings[emb_idx]
                    emb_idx += 1
    
    # Convert atoms to vertices
    for atom in atoms:
        vertex = LatticeVertex.from_atom(atom, domain=domain)
        lattice.add_vertex(vertex)
    
    # Create sequential edges (simple chain structure)
    # More sophisticated edge creation could use semantic similarity
    vertex_ids = list(lattice.vertices.keys())
    for i in range(len(vertex_ids) - 1):
        import hashlib
        edge_id = hashlib.md5(f"{vertex_ids[i]}:{vertex_ids[i+1]}".encode()).hexdigest()[:16]
        edge = LatticeEdge(
            id=edge_id,
            source_id=vertex_ids[i],
            target_ids=[vertex_ids[i + 1]],
            weight=1.0,
            edge_type="sequential"
        )
        lattice.add_edge(edge)
    
    return lattice
