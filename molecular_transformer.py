"""
Molecular Transformer Model for Forward Reaction Prediction
Implementation of a simplified transformer-based model for chemical reaction prediction.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

from bayesian_retrosynthesis import Molecule, ForwardModel

class MolecularTransformer(ForwardModel):
    """
    Simplified Molecular Transformer model for forward reaction prediction.
    
    This implements a basic version of the transformer architecture adapted
    for chemical reaction prediction, similar to the models referenced in the paper.
    """
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 6):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initialize vocabulary and tokenizer
        self.vocab = self._build_vocabulary()
        self.tokenizer = SMILESTokenizer(self.vocab)
        
        # Reaction templates based on common organic reactions
        self.reaction_templates = self._load_reaction_templates()
        
        # Pre-trained weights (simulated)
        self.weights = self._initialize_weights()
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary for SMILES tokenization."""
        # Common SMILES tokens
        tokens = [
            '<PAD>', '<START>', '<END>', '<UNK>',
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's', 'p',  # Aromatic
            '(', ')', '[', ']', '=', '#', '+', '-',
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '@', '@@', '/', '\\', '.', '>>'
        ]
        
        # Add more complex tokens
        for i in range(10, 100):
            tokens.append(f'C{i}')  # Carbon chains
        
        return {token: idx for idx, token in enumerate(tokens)}
    
    def _load_reaction_templates(self) -> List[Dict]:
        """Load reaction templates for different reaction types."""
        return [
            {
                "name": "Suzuki_coupling",
                "pattern": r"(\w+)Br\.(\w+)B\(OH\)2",
                "product_template": r"\1\2",
                "probability": 0.85,
                "conditions": ["Pd catalyst", "base"]
            },
            {
                "name": "Grignard_reaction",
                "pattern": r"(\w+)Br\.(\w+)=O",
                "product_template": r"\1C(\2)O",
                "probability": 0.75,
                "conditions": ["Mg", "ether"]
            },
            {
                "name": "Aldol_condensation",
                "pattern": r"(\w+)C=O\.(\w+)C=O",
                "product_template": r"\1C(O)C\2C=O",
                "probability": 0.70,
                "conditions": ["base", "heat"]
            },
            {
                "name": "Diels_Alder",
                "pattern": r"(\w+)C=CC=C(\w+)\.(\w+)C=C(\w+)",
                "product_template": r"\1C1CC(\3)C(\4)CC1\2",
                "probability": 0.90,
                "conditions": ["heat"]
            },
            {
                "name": "Nucleophilic_substitution",
                "pattern": r"(\w+)Cl\.(\w+)OH",
                "product_template": r"\1O\2",
                "probability": 0.65,
                "conditions": ["base"]
            },
            {
                "name": "Oxidation",
                "pattern": r"(\w+)OH",
                "product_template": r"\1=O",
                "probability": 0.80,
                "conditions": ["oxidizing agent"]
            },
            {
                "name": "Reduction",
                "pattern": r"(\w+)=O",
                "product_template": r"\1OH",
                "probability": 0.75,
                "conditions": ["reducing agent"]
            },
            {
                "name": "Esterification",
                "pattern": r"(\w+)COOH\.(\w+)OH",
                "product_template": r"\1COO\2",
                "probability": 0.70,
                "conditions": ["acid catalyst"]
            }
        ]
    
    def _initialize_weights(self) -> Dict:
        """Initialize model weights (simulated)."""
        return {
            "embedding": np.random.randn(self.vocab_size, self.embedding_dim) * 0.1,
            "attention": [np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1 
                         for _ in range(self.num_layers)],
            "output": np.random.randn(self.embedding_dim, self.vocab_size) * 0.1
        }
    
    def predict(self, reactants: List[Molecule]) -> List[Tuple[List[Molecule], float]]:
        """
        Predict products from reactants using the molecular transformer.
        
        Args:
            reactants: List of reactant molecules
            
        Returns:
            List of (products, probability) tuples
        """
        if not reactants:
            return []
        
        # Convert reactants to SMILES string
        reactant_smiles = '.'.join([mol.smiles for mol in reactants])
        
        # Apply reaction templates
        predictions = []
        
        for template in self.reaction_templates:
            products = self._apply_template(reactant_smiles, template)
            if products:
                # Calculate probability using transformer attention
                prob = self._calculate_probability(reactant_smiles, products, template)
                
                # Convert to Molecule objects
                product_molecules = [Molecule(smiles) for smiles in products.split('.')]
                predictions.append((product_molecules, prob))
        
        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Add some noise for diversity
        for i in range(len(predictions)):
            products, prob = predictions[i]
            noise = random.uniform(0.9, 1.1)
            predictions[i] = (products, min(prob * noise, 1.0))
        
        return predictions[:10]  # Return top 10 predictions
    
    def _apply_template(self, reactants: str, template: Dict) -> Optional[str]:
        """Apply reaction template to generate products."""
        pattern = template["pattern"]
        product_template = template["product_template"]
        
        # Simple pattern matching
        match = re.search(pattern, reactants)
        if match:
            try:
                # Apply template transformation
                product = re.sub(pattern, product_template, reactants)
                return product
            except:
                pass
        
        # Fallback: apply simple transformations
        return self._apply_simple_transformation(reactants, template)
    
    def _apply_simple_transformation(self, reactants: str, template: Dict) -> Optional[str]:
        """Apply simple chemical transformations."""
        name = template["name"]
        
        if name == "Oxidation" and "OH" in reactants:
            return reactants.replace("OH", "=O")
        
        elif name == "Reduction" and "=O" in reactants:
            return reactants.replace("=O", "OH")
        
        elif name == "Nucleophilic_substitution":
            if "Cl" in reactants and "OH" in reactants:
                # Remove Cl and OH, add O bridge
                result = reactants.replace("Cl", "").replace(".OH", "O")
                return result
        
        elif name == "Esterification":
            if "COOH" in reactants and "OH" in reactants:
                # Form ester bond
                result = reactants.replace("COOH", "COO").replace(".OH", "")
                return result
        
        # Default: return modified reactants
        if len(reactants) > 5:
            return reactants + "C"  # Add carbon
        
        return None
    
    def _calculate_probability(self, reactants: str, products: str, 
                             template: Dict) -> float:
        """Calculate reaction probability using transformer attention."""
        # Tokenize input
        reactant_tokens = self.tokenizer.tokenize(reactants)
        product_tokens = self.tokenizer.tokenize(products)
        
        # Simulate transformer attention calculation
        attention_score = self._simulate_attention(reactant_tokens, product_tokens)
        
        # Combine with template probability
        template_prob = template["probability"]
        
        # Add molecular complexity factor
        complexity_factor = self._calculate_complexity_factor(reactants, products)
        
        # Final probability
        prob = template_prob * attention_score * complexity_factor
        
        # Add some randomness for diversity
        noise = random.uniform(0.8, 1.2)
        
        return min(prob * noise, 1.0)
    
    def _simulate_attention(self, reactant_tokens: List[int], 
                          product_tokens: List[int]) -> float:
        """Simulate transformer attention mechanism."""
        if not reactant_tokens or not product_tokens:
            return 0.1
        
        # Simple attention simulation based on token overlap
        reactant_set = set(reactant_tokens)
        product_set = set(product_tokens)
        
        # Jaccard similarity
        intersection = len(reactant_set & product_set)
        union = len(reactant_set | product_set)
        
        if union == 0:
            return 0.1
        
        similarity = intersection / union
        
        # Transform to attention score
        attention = 0.5 + 0.5 * similarity
        
        return attention
    
    def _calculate_complexity_factor(self, reactants: str, products: str) -> float:
        """Calculate complexity factor for the transformation."""
        # Count structural features
        reactant_complexity = self._count_features(reactants)
        product_complexity = self._count_features(products)
        
        # Prefer reactions that don't drastically change complexity
        complexity_diff = abs(product_complexity - reactant_complexity)
        
        # Normalize
        factor = 1.0 / (1.0 + complexity_diff / 10.0)
        
        return factor
    
    def _count_features(self, smiles: str) -> float:
        """Count structural features in SMILES."""
        features = 0.0
        features += smiles.count('(') * 2  # Branching
        features += smiles.count('=') * 1.5  # Double bonds
        features += smiles.count('#') * 2  # Triple bonds
        features += smiles.count('[') * 3  # Special atoms
        features += len(re.findall(r'[A-Z]', smiles))  # Atoms
        
        return features

class SMILESTokenizer:
    """Tokenizer for SMILES strings."""
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
    
    def tokenize(self, smiles: str) -> List[int]:
        """Tokenize SMILES string into token IDs."""
        tokens = []
        i = 0
        
        while i < len(smiles):
            # Try to match longest token first
            matched = False
            
            for length in range(min(5, len(smiles) - i), 0, -1):
                token = smiles[i:i+length]
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Unknown token
                tokens.append(self.vocab.get('<UNK>', 0))
                i += 1
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to SMILES string."""
        tokens = [self.reverse_vocab.get(tid, '<UNK>') for tid in token_ids]
        return ''.join(tokens)

class ReactionDatabase:
    """Database of known chemical reactions for training and validation."""
    
    def __init__(self):
        self.reactions = []
        self._load_sample_reactions()
    
    def _load_sample_reactions(self):
        """Load sample reactions for demonstration."""
        sample_reactions = [
            {
                "reactants": ["CCO", "CC(=O)Cl"],
                "products": ["CC(=O)OCC"],
                "reaction_type": "esterification",
                "yield": 0.85
            },
            {
                "reactants": ["CC(=O)C", "NaBH4"],
                "products": ["CC(O)C"],
                "reaction_type": "reduction",
                "yield": 0.90
            },
            {
                "reactants": ["CCO", "[O]"],
                "products": ["CC=O"],
                "reaction_type": "oxidation",
                "yield": 0.75
            },
            {
                "reactants": ["C=C", "Br2"],
                "products": ["CBrCBr"],
                "reaction_type": "addition",
                "yield": 0.95
            },
            {
                "reactants": ["CCl", "NaOH"],
                "products": ["CO"],
                "reaction_type": "substitution",
                "yield": 0.70
            }
        ]
        
        for rxn_data in sample_reactions:
            reactants = [Molecule(smiles) for smiles in rxn_data["reactants"]]
            products = [Molecule(smiles) for smiles in rxn_data["products"]]
            
            reaction = {
                "reactants": reactants,
                "products": products,
                "type": rxn_data["reaction_type"],
                "yield": rxn_data["yield"]
            }
            
            self.reactions.append(reaction)
    
    def add_reaction(self, reactants: List[str], products: List[str], 
                    reaction_type: str, yield_value: float = 1.0):
        """Add a new reaction to the database."""
        reactant_mols = [Molecule(smiles) for smiles in reactants]
        product_mols = [Molecule(smiles) for smiles in products]
        
        reaction = {
            "reactants": reactant_mols,
            "products": product_mols,
            "type": reaction_type,
            "yield": yield_value
        }
        
        self.reactions.append(reaction)
    
    def get_reactions_by_type(self, reaction_type: str) -> List[Dict]:
        """Get all reactions of a specific type."""
        return [rxn for rxn in self.reactions if rxn["type"] == reaction_type]
    
    def get_similar_reactions(self, target_reactants: List[Molecule]) -> List[Dict]:
        """Find reactions with similar reactants."""
        similar = []
        target_smiles = set(mol.smiles for mol in target_reactants)
        
        for reaction in self.reactions:
            rxn_smiles = set(mol.smiles for mol in reaction["reactants"])
            
            # Calculate similarity (Jaccard index)
            intersection = len(target_smiles & rxn_smiles)
            union = len(target_smiles | rxn_smiles)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.3:  # Threshold for similarity
                    similar.append({
                        "reaction": reaction,
                        "similarity": similarity
                    })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)

def main():
    """Demonstration of the Molecular Transformer model."""
    # Create molecular transformer
    transformer = MolecularTransformer()
    
    # Create reaction database
    db = ReactionDatabase()
    
    # Test molecules
    test_reactants = [
        [Molecule("CCO")],  # Ethanol
        [Molecule("CC(=O)C")],  # Acetone
        [Molecule("CCl"), Molecule("NaOH")],  # Chloroethane + base
        [Molecule("CC(=O)Cl"), Molecule("CCO")]  # Acetyl chloride + ethanol
    ]
    
    print("Molecular Transformer Predictions:")
    print("=" * 50)
    
    for i, reactants in enumerate(test_reactants, 1):
        reactant_smiles = '.'.join([mol.smiles for mol in reactants])
        print(f"\nTest {i}: {reactant_smiles}")
        
        predictions = transformer.predict(reactants)
        
        print(f"Found {len(predictions)} predictions:")
        
        for j, (products, prob) in enumerate(predictions[:3], 1):
            product_smiles = '.'.join([mol.smiles for mol in products])
            print(f"  {j}. {product_smiles} (probability: {prob:.3f})")
        
        # Find similar reactions in database
        similar = db.get_similar_reactions(reactants)
        if similar:
            print(f"  Similar known reactions:")
            for sim_data in similar[:2]:
                rxn = sim_data["reaction"]
                similarity = sim_data["similarity"]
                rxn_reactants = '.'.join([mol.smiles for mol in rxn["reactants"]])
                rxn_products = '.'.join([mol.smiles for mol in rxn["products"]])
                print(f"    {rxn_reactants} >> {rxn_products} (similarity: {similarity:.3f})")

if __name__ == "__main__":
    main()
