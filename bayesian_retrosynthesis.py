"""
Bayesian Algorithm for Retrosynthesis
Implementation based on the paper: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00320

This module implements a Bayesian approach to retrosynthesis prediction using
Monte Carlo sampling and conditional probability distributions.
"""

import numpy as np
import random
import sys
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Molecule:
    """Represents a molecule with SMILES notation and properties."""
    smiles: str
    molecular_weight: float = 0.0
    complexity_score: float = 0.0
    
    def __post_init__(self):
        if self.molecular_weight == 0.0:
            self.molecular_weight = self._estimate_molecular_weight()
        if self.complexity_score == 0.0:
            self.complexity_score = self._estimate_complexity()
    
    def _estimate_molecular_weight(self) -> float:
        """Simple estimation of molecular weight based on SMILES length."""
        return len(self.smiles) * 12.0  # Rough approximation
    
    def _estimate_complexity(self) -> float:
        """Estimate molecular complexity based on structural features."""
        complexity = 0.0
        complexity += self.smiles.count('(') * 2  # Branching
        complexity += self.smiles.count('[') * 3  # Special atoms
        complexity += self.smiles.count('=') * 1.5  # Double bonds
        complexity += self.smiles.count('#') * 2  # Triple bonds
        return complexity

@dataclass
class Reaction:
    """Represents a chemical reaction."""
    reactants: List[Molecule]
    products: List[Molecule]
    reaction_type: str
    probability: float = 1.0
    
    def __str__(self):
        reactant_smiles = '.'.join([mol.smiles for mol in self.reactants])
        product_smiles = '.'.join([mol.smiles for mol in self.products])
        return f"{reactant_smiles}>>{product_smiles}"

class ForwardModel(ABC):
    """Abstract base class for forward reaction prediction models."""
    
    @abstractmethod
    def predict(self, reactants: List[Molecule]) -> List[Tuple[List[Molecule], float]]:
        """Predict products from reactants with probabilities."""
        pass

class SimpleForwardModel(ForwardModel):
    """Simple forward model for demonstration purposes."""
    
    def __init__(self):
        # Simple reaction templates for demonstration
        self.reaction_templates = [
            {"name": "oxidation", "pattern": "alcohol_to_aldehyde", "prob": 0.8},
            {"name": "reduction", "pattern": "ketone_to_alcohol", "prob": 0.7},
            {"name": "substitution", "pattern": "halide_substitution", "prob": 0.6},
            {"name": "addition", "pattern": "alkene_addition", "prob": 0.9},
        ]
    
    def predict(self, reactants: List[Molecule]) -> List[Tuple[List[Molecule], float]]:
        """Simple prediction based on molecular complexity."""
        predictions = []
        
        for template in self.reaction_templates:
            # Simple heuristic: generate products based on reactant complexity
            product_smiles = self._apply_template(reactants, template)
            if product_smiles:
                product = Molecule(product_smiles)
                probability = template["prob"] * random.uniform(0.5, 1.0)
                predictions.append(([product], probability))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    
    def _apply_template(self, reactants: List[Molecule], template: Dict) -> Optional[str]:
        """Apply reaction template to generate product SMILES."""
        if not reactants:
            return None
        
        # Simple transformation based on template type
        base_smiles = reactants[0].smiles
        
        if template["name"] == "oxidation":
            return base_smiles + "=O"  # Add carbonyl
        elif template["name"] == "reduction":
            return base_smiles.replace("=O", "O")  # Reduce carbonyl
        elif template["name"] == "substitution":
            return base_smiles.replace("Cl", "OH")  # Substitute halide
        elif template["name"] == "addition":
            return base_smiles + "CC"  # Add carbon chain
        
        return base_smiles

class BayesianRetrosynthesis:
    """
    Main class implementing the Bayesian retrosynthesis algorithm.
    
    Uses Monte Carlo sampling to explore retrosynthetic pathways based on
    Bayesian conditional probabilities P(reactants|product).
    """
    
    def __init__(self, forward_model: ForwardModel, max_depth: int = 5, 
                 num_samples: int = 1000, temperature: float = 1.0):
        self.forward_model = forward_model
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.temperature = temperature
        self.reaction_database = []
        
    def add_reaction(self, reaction: Reaction):
        """Add a reaction to the database."""
        self.reaction_database.append(reaction)
    
    def predict_retrosynthesis(self, target_molecule: Molecule, 
                             num_pathways: int = 10) -> List[List[Reaction]]:
        """
        Predict retrosynthetic pathways for a target molecule using Bayesian sampling.
        
        Args:
            target_molecule: The target molecule to synthesize
            num_pathways: Number of pathways to return
            
        Returns:
            List of retrosynthetic pathways (each pathway is a list of reactions)
        """
        logger.info(f"Starting retrosynthesis prediction for: {target_molecule.smiles}")
        
        pathways = []
        
        for _ in range(self.num_samples):
            pathway = self._sample_pathway(target_molecule)
            if pathway and self._is_valid_pathway(pathway):
                pathways.append(pathway)
        
        # Score and rank pathways
        scored_pathways = [(path, self._score_pathway(path)) for path in pathways]
        scored_pathways.sort(key=lambda x: x[1], reverse=True)
        
        return [path for path, score in scored_pathways[:num_pathways]]
    
    def _sample_pathway(self, target: Molecule, depth: int = 0) -> List[Reaction]:
        """Sample a single retrosynthetic pathway using Monte Carlo."""
        if depth >= self.max_depth:
            return []
        
        # Check if target is a simple starting material
        if self._is_starting_material(target):
            return []
        
        # Sample potential precursors using Bayesian inference
        precursors = self._sample_precursors(target)
        
        if not precursors:
            return []
        
        # Create reaction
        reaction = Reaction(
            reactants=precursors,
            products=[target],
            reaction_type="retrosynthetic_step",
            probability=self._calculate_reaction_probability(precursors, target)
        )
        
        # Recursively sample pathways for precursors
        pathway = [reaction]
        for precursor in precursors:
            sub_pathway = self._sample_pathway(precursor, depth + 1)
            pathway.extend(sub_pathway)
        
        return pathway
    
    def _sample_precursors(self, target: Molecule) -> List[Molecule]:
        """
        Sample potential precursors using Bayesian conditional probability P(S|Y).
        
        This implements the core Bayesian inference: given a target product Y,
        what are the most likely starting materials S?
        """
        candidates = []
        
        # Generate candidate precursors based on retrosynthetic rules
        for rule in self._get_retrosynthetic_rules():
            precursor_smiles = self._apply_retrosynthetic_rule(target.smiles, rule)
            if precursor_smiles:
                precursor = Molecule(precursor_smiles)
                
                # Calculate Bayesian probability P(precursor|target)
                prob = self._calculate_bayesian_probability(precursor, target)
                candidates.append((precursor, prob))
        
        # Sample from candidates using temperature-scaled probabilities
        if not candidates:
            return []
        
        # Normalize probabilities
        total_prob = sum(prob for _, prob in candidates)
        if total_prob == 0:
            return []
        
        normalized_probs = [prob / total_prob for _, prob in candidates]
        
        # Temperature scaling for exploration vs exploitation
        scaled_probs = np.array(normalized_probs) ** (1.0 / self.temperature)
        scaled_probs /= scaled_probs.sum()
        
        # Sample precursors
        num_precursors = min(2, len(candidates))  # Usually 1-2 precursors
        selected_indices = np.random.choice(
            len(candidates), 
            size=num_precursors, 
            replace=False, 
            p=scaled_probs
        )
        
        return [candidates[i][0] for i in selected_indices]
    
    def _calculate_bayesian_probability(self, precursor: Molecule, target: Molecule) -> float:
        """
        Calculate P(precursor|target) using Bayes' theorem:
        P(S|Y) = P(Y|S) * P(S) / P(Y)
        
        Where:
        - P(Y|S) is the forward reaction probability (from forward model)
        - P(S) is the prior probability of the precursor
        - P(Y) is the prior probability of the target
        """
        # Forward probability P(Y|S)
        forward_predictions = self.forward_model.predict([precursor])
        forward_prob = 0.0
        
        for products, prob in forward_predictions:
            if any(prod.smiles == target.smiles for prod in products):
                forward_prob = prob
                break
        
        # Prior probabilities based on molecular complexity
        prior_precursor = self._calculate_prior_probability(precursor)
        prior_target = self._calculate_prior_probability(target)
        
        # Avoid division by zero
        if prior_target == 0:
            return 0.0
        
        # Bayes' theorem
        bayesian_prob = (forward_prob * prior_precursor) / prior_target
        
        return min(bayesian_prob, 1.0)  # Cap at 1.0
    
    def _calculate_prior_probability(self, molecule: Molecule) -> float:
        """Calculate prior probability based on molecular properties."""
        # Simpler molecules have higher prior probability
        complexity_factor = 1.0 / (1.0 + molecule.complexity_score / 10.0)
        
        # Commercial availability factor (simplified)
        availability_factor = 0.8 if len(molecule.smiles) < 20 else 0.3
        
        return complexity_factor * availability_factor
    
    def _get_retrosynthetic_rules(self) -> List[Dict]:
        """Get retrosynthetic transformation rules."""
        return [
            {"name": "functional_group_interconversion", "type": "FGI"},
            {"name": "carbon_carbon_bond_formation", "type": "C-C"},
            {"name": "carbon_heteroatom_bond_formation", "type": "C-X"},
            {"name": "ring_formation", "type": "cyclization"},
            {"name": "protecting_group_strategy", "type": "protection"},
        ]
    
    def _apply_retrosynthetic_rule(self, target_smiles: str, rule: Dict) -> Optional[str]:
        """Apply retrosynthetic rule to generate precursor SMILES."""
        # Simplified rule application
        if rule["type"] == "FGI":
            # Functional group interconversion
            if "=O" in target_smiles:
                return target_smiles.replace("=O", "O")  # Carbonyl to alcohol
            elif "O" in target_smiles:
                return target_smiles.replace("O", "")  # Remove oxygen
        
        elif rule["type"] == "C-C":
            # Carbon-carbon bond disconnection
            if len(target_smiles) > 5:
                # Simple disconnection
                mid_point = len(target_smiles) // 2
                return target_smiles[:mid_point]
        
        elif rule["type"] == "C-X":
            # Carbon-heteroatom disconnection
            if "N" in target_smiles:
                return target_smiles.replace("N", "")
        
        # Default: return a simplified version
        if len(target_smiles) > 3:
            return target_smiles[:-2]
        
        return None
    
    def _calculate_reaction_probability(self, reactants: List[Molecule], 
                                     product: Molecule) -> float:
        """Calculate the probability of a reaction occurring."""
        # Use forward model to estimate probability
        if not reactants:
            return 0.0
        
        predictions = self.forward_model.predict(reactants)
        
        for products, prob in predictions:
            if any(prod.smiles == product.smiles for prod in products):
                return prob
        
        # Default probability based on molecular complexity
        reactant_complexity = sum(mol.complexity_score for mol in reactants)
        product_complexity = product.complexity_score
        
        # Reactions that reduce complexity are more favorable
        complexity_ratio = reactant_complexity / (product_complexity + 1.0)
        return min(complexity_ratio / 10.0, 1.0)
    
    def _is_starting_material(self, molecule: Molecule) -> bool:
        """Check if a molecule is a simple starting material."""
        # Simple heuristics for starting materials
        return (
            len(molecule.smiles) < 10 or
            molecule.complexity_score < 5.0 or
            molecule.smiles in ["CCO", "CC", "C", "CO", "CCl", "CBr"]
        )
    
    def _is_valid_pathway(self, pathway: List[Reaction]) -> bool:
        """Check if a retrosynthetic pathway is valid."""
        if not pathway:
            return False
        
        # Check that all reactions are chemically reasonable
        for reaction in pathway:
            if reaction.probability < 0.1:  # Very low probability reactions
                return False
        
        return True
    
    def _score_pathway(self, pathway: List[Reaction]) -> float:
        """Score a retrosynthetic pathway based on multiple criteria."""
        if not pathway:
            return 0.0
        
        # Probability score (product of all reaction probabilities)
        prob_score = 1.0
        for reaction in pathway:
            prob_score *= reaction.probability
        
        # Length penalty (shorter pathways are better)
        length_penalty = 1.0 / (1.0 + len(pathway) / 5.0)
        
        # Complexity reduction score
        complexity_score = self._calculate_complexity_reduction(pathway)
        
        return prob_score * length_penalty * complexity_score
    
    def _calculate_complexity_reduction(self, pathway: List[Reaction]) -> float:
        """Calculate how much complexity is reduced in the pathway."""
        if not pathway:
            return 0.0
        
        # Get starting and ending complexities
        start_complexity = sum(
            mol.complexity_score for mol in pathway[0].products
        )
        
        end_complexity = sum(
            mol.complexity_score for mol in pathway[-1].reactants
        )
        
        if start_complexity == 0:
            return 1.0
        
        reduction_ratio = (start_complexity - end_complexity) / start_complexity
        return max(reduction_ratio, 0.1)  # Minimum score

def main():
    """Demonstration of the Bayesian retrosynthesis algorithm."""
    parser = argparse.ArgumentParser(
        description='Bayesian Retrosynthesis Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bayesian_retrosynthesis.py
  python bayesian_retrosynthesis.py --smiles "CC(=O)OCC"
  python bayesian_retrosynthesis.py --smiles "CC(C)(C)OC(=O)NC(Cc1ccccc1)C(O)Cn1nnc2ccccc21" --max-depth 4 --num-samples 200
  python bayesian_retrosynthesis.py --smiles "CCO" --num-pathways 3 --temperature 1.5
        """
    )
    
    parser.add_argument(
        '--smiles', '-s',
        type=str,
        default="CC(=O)OCC",
        help='Target molecule SMILES string (default: CC(=O)OCC - ethyl acetate)'
    )
    
    parser.add_argument(
        '--max-depth', '-d',
        type=int,
        default=3,
        help='Maximum depth for retrosynthetic search (default: 3)'
    )
    
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=100,
        help='Number of Monte Carlo samples (default: 100)'
    )
    
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=1.0,
        help='Temperature parameter for sampling (default: 1.0)'
    )
    
    parser.add_argument(
        '--num-pathways', '-p',
        type=int,
        default=5,
        help='Number of pathways to return (default: 5)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create forward model
    forward_model = SimpleForwardModel()
    
    # Create Bayesian retrosynthesis predictor
    predictor = BayesianRetrosynthesis(
        forward_model=forward_model,
        max_depth=args.max_depth,
        num_samples=args.num_samples,
        temperature=args.temperature
    )
    
    # Create target molecule
    try:
        target = Molecule(args.smiles)
    except Exception as e:
        print(f"Error creating molecule from SMILES '{args.smiles}': {e}")
        sys.exit(1)
    
    print("=" * 70)
    print("Bayesian Retrosynthesis Prediction")
    print("=" * 70)
    print(f"Target molecule: {target.smiles}")
    print(f"Molecular weight: {target.molecular_weight:.2f}")
    print(f"Complexity score: {target.complexity_score:.2f}")
    print(f"Parameters: max_depth={args.max_depth}, num_samples={args.num_samples}, temperature={args.temperature}")
    print()
    
    # Predict retrosynthetic pathways
    import time
    start_time = time.time()
    pathways = predictor.predict_retrosynthesis(target, num_pathways=args.num_pathways)
    end_time = time.time()
    
    print(f"Found {len(pathways)} retrosynthetic pathways in {end_time - start_time:.2f} seconds:")
    print()
    
    if not pathways:
        print("No valid pathways found. Try increasing num_samples or adjusting other parameters.")
        return
    
    for i, pathway in enumerate(pathways, 1):
        print(f"Pathway {i}:")
        score = predictor._score_pathway(pathway)
        print(f"  Score: {score:.4f}")
        print(f"  Steps: {len(pathway)}")
        
        for j, reaction in enumerate(pathway):
            reactant_smiles = '.'.join([mol.smiles for mol in reaction.reactants])
            product_smiles = '.'.join([mol.smiles for mol in reaction.products])
            print(f"  Step {j+1}: {reactant_smiles} >> {product_smiles}")
            print(f"    Type: {reaction.reaction_type}")
            print(f"    Probability: {reaction.probability:.3f}")
            
            # Show reactant details
            if args.verbose:
                for k, reactant in enumerate(reaction.reactants):
                    print(f"      Reactant {k+1}: {reactant.smiles} (MW: {reactant.molecular_weight:.1f}, Complexity: {reactant.complexity_score:.1f})")
        print()
    
    # Summary statistics
    if pathways:
        scores = [predictor._score_pathway(p) for p in pathways]
        pathway_lengths = [len(p) for p in pathways]
        
        print("Summary Statistics:")
        print(f"  Average pathway score: {np.mean(scores):.4f}")
        print(f"  Best pathway score: {max(scores):.4f}")
        print(f"  Average pathway length: {np.mean(pathway_lengths):.1f} steps")
        print(f"  Shortest pathway: {min(pathway_lengths)} steps")
        print(f"  Longest pathway: {max(pathway_lengths)} steps")

if __name__ == "__main__":
    main()
