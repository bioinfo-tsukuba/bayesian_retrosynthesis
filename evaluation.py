"""
Evaluation and Benchmarking for Bayesian Retrosynthesis
Implementation of evaluation metrics and benchmarking against other methods.
"""

import numpy as np
import time
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from bayesian_retrosynthesis import BayesianRetrosynthesis, Molecule, Reaction
from molecular_transformer import MolecularTransformer, ReactionDatabase

@dataclass
class EvaluationResult:
    """Results from evaluating a retrosynthesis method."""
    method_name: str
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    top_10_accuracy: float
    average_time: float
    total_predictions: int
    valid_pathways: int

class RetrosynthesisEvaluator:
    """
    Evaluator for retrosynthesis prediction methods.
    
    Implements the evaluation metrics used in the paper to compare
    different approaches including the Bayesian method.
    """
    
    def __init__(self):
        self.test_molecules = self._load_test_molecules()
        self.ground_truth = self._load_ground_truth()
        
    def _load_test_molecules(self) -> List[Molecule]:
        """Load test molecules for evaluation."""
        # Test molecules from common synthetic targets
        test_smiles = [
            "CC(=O)OCC",  # Ethyl acetate
            "CC(C)C(=O)O",  # Isobutyric acid
            "CCc1ccccc1",  # Ethylbenzene
            "CC(=O)c1ccccc1",  # Acetophenone
            "CCN(CC)CC",  # Triethylamine
            "CC(C)(C)OC(=O)Nc1ccccc1",  # Boc-aniline
            "COc1ccc(C=O)cc1",  # p-Anisaldehyde
            "CC(=O)Nc1ccccc1",  # Acetanilide
            "CCOc1ccc(C(=O)O)cc1",  # Ethyl p-hydroxybenzoate
            "CC1=CC(=O)CCC1",  # 3-Methyl-2-cyclohexen-1-one
            "CC(C)C(=O)Cl",  # Isobutyryl chloride
            "CCc1ccc(O)cc1",  # 4-Ethylphenol
            "CC(=O)OC(C)C",  # Isopropyl acetate
            "CCN(C)C(=O)c1ccccc1",  # N,N-Diethylbenzamide
            "CC1CCC(=O)CC1"  # 4-Methylcyclohexanone
        ]
        
        return [Molecule(smiles) for smiles in test_smiles]
    
    def _load_ground_truth(self) -> Dict[str, List[List[str]]]:
        """Load ground truth retrosynthetic pathways."""
        # Simplified ground truth pathways
        ground_truth = {
            "CC(=O)OCC": [  # Ethyl acetate
                ["CC(=O)Cl", "CCO"],  # Acetyl chloride + ethanol
                ["CC(=O)O", "CCO"]    # Acetic acid + ethanol
            ],
            "CC(C)C(=O)O": [  # Isobutyric acid
                ["CC(C)C(=O)Cl"],     # From acid chloride
                ["CC(C)CN"]           # From nitrile
            ],
            "CCc1ccccc1": [  # Ethylbenzene
                ["c1ccccc1", "CCCl"], # Benzene + ethyl chloride
                ["c1ccccc1C=O", "CC"] # Benzaldehyde + methyl
            ],
            "CC(=O)c1ccccc1": [  # Acetophenone
                ["c1ccccc1", "CC(=O)Cl"], # Benzene + acetyl chloride
                ["c1ccccc1C=O", "CC"]     # Benzaldehyde + methyl
            ],
            "CCN(CC)CC": [  # Triethylamine
                ["CCN", "CCCl"],      # Diethylamine + ethyl chloride
                ["N", "CCCl"]         # Ammonia + ethyl chloride
            ]
        }
        
        return ground_truth
    
    def evaluate_method(self, method, method_name: str, 
                       num_predictions: int = 10) -> EvaluationResult:
        """
        Evaluate a retrosynthesis prediction method.
        
        Args:
            method: The prediction method to evaluate
            method_name: Name of the method
            num_predictions: Number of predictions to generate per molecule
            
        Returns:
            EvaluationResult with accuracy metrics
        """
        print(f"Evaluating {method_name}...")
        
        top_1_correct = 0
        top_3_correct = 0
        top_5_correct = 0
        top_10_correct = 0
        total_time = 0.0
        total_predictions = 0
        valid_pathways = 0
        
        for molecule in self.test_molecules:
            start_time = time.time()
            
            # Get predictions
            if hasattr(method, 'predict_retrosynthesis'):
                # Bayesian method
                pathways = method.predict_retrosynthesis(molecule, num_predictions)
                predictions = self._pathways_to_predictions(pathways)
            else:
                # Forward model
                predictions = method.predict([molecule])
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            if not predictions:
                continue
            
            total_predictions += len(predictions)
            
            # Check against ground truth
            ground_truth = self.ground_truth.get(molecule.smiles, [])
            
            if ground_truth:
                # Check top-k accuracy
                for k, pred in enumerate(predictions[:10]):
                    if self._is_correct_prediction(pred, ground_truth):
                        if k == 0:
                            top_1_correct += 1
                        if k < 3:
                            top_3_correct += 1
                        if k < 5:
                            top_5_correct += 1
                        if k < 10:
                            top_10_correct += 1
                        break
            
            # Count valid pathways
            valid_pathways += len([p for p in predictions if self._is_valid_prediction(p)])
        
        # Calculate accuracies
        num_test_molecules = len([mol for mol in self.test_molecules 
                                if mol.smiles in self.ground_truth])
        
        if num_test_molecules == 0:
            num_test_molecules = len(self.test_molecules)
        
        return EvaluationResult(
            method_name=method_name,
            top_1_accuracy=top_1_correct / num_test_molecules,
            top_3_accuracy=top_3_correct / num_test_molecules,
            top_5_accuracy=top_5_correct / num_test_molecules,
            top_10_accuracy=top_10_correct / num_test_molecules,
            average_time=total_time / len(self.test_molecules),
            total_predictions=total_predictions,
            valid_pathways=valid_pathways
        )
    
    def _pathways_to_predictions(self, pathways: List[List[Reaction]]) -> List[Tuple]:
        """Convert pathways to prediction format."""
        predictions = []
        
        for pathway in pathways:
            if pathway:
                # Get starting materials from the pathway
                starting_materials = []
                for reaction in pathway:
                    starting_materials.extend(reaction.reactants)
                
                # Calculate pathway probability
                prob = 1.0
                for reaction in pathway:
                    prob *= reaction.probability
                
                predictions.append((starting_materials, prob))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    
    def _is_correct_prediction(self, prediction: Tuple, ground_truth: List[List[str]]) -> bool:
        """Check if a prediction matches ground truth."""
        pred_molecules, _ = prediction
        pred_smiles = set(mol.smiles for mol in pred_molecules)
        
        for gt_pathway in ground_truth:
            gt_smiles = set(gt_pathway)
            
            # Check if prediction matches any ground truth pathway
            if pred_smiles == gt_smiles or pred_smiles.issubset(gt_smiles):
                return True
        
        return False
    
    def _is_valid_prediction(self, prediction: Tuple) -> bool:
        """Check if a prediction is chemically valid."""
        molecules, prob = prediction
        
        # Basic validity checks
        if prob < 0.1:  # Very low probability
            return False
        
        if not molecules:
            return False
        
        # Check for reasonable starting materials
        for mol in molecules:
            if len(mol.smiles) > 50:  # Too complex
                return False
            if mol.complexity_score > 20:  # Too complex
                return False
        
        return True
    
    def compare_methods(self, methods: List[Tuple], save_results: bool = True) -> Dict:
        """
        Compare multiple retrosynthesis methods.
        
        Args:
            methods: List of (method, name) tuples
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with comparison results
        """
        results = []
        
        for method, name in methods:
            result = self.evaluate_method(method, name)
            results.append(result)
        
        # Create comparison table
        comparison = self._create_comparison_table(results)
        
        if save_results:
            self._save_results(results, comparison)
        
        return {
            "individual_results": results,
            "comparison_table": comparison
        }
    
    def _create_comparison_table(self, results: List[EvaluationResult]) -> Dict:
        """Create comparison table similar to the paper."""
        table = {
            "method": [],
            "top_1": [],
            "top_3": [],
            "top_5": [],
            "top_10": [],
            "avg_time": [],
            "valid_pathways": []
        }
        
        for result in results:
            table["method"].append(result.method_name)
            table["top_1"].append(f"{result.top_1_accuracy:.3f}")
            table["top_3"].append(f"{result.top_3_accuracy:.3f}")
            table["top_5"].append(f"{result.top_5_accuracy:.3f}")
            table["top_10"].append(f"{result.top_10_accuracy:.3f}")
            table["avg_time"].append(f"{result.average_time:.3f}s")
            table["valid_pathways"].append(f"{result.valid_pathways}/{result.total_predictions}")
        
        return table
    
    def _save_results(self, results: List[EvaluationResult], comparison: Dict):
        """Save evaluation results to files."""
        # Save detailed results
        results_data = []
        for result in results:
            results_data.append({
                "method_name": result.method_name,
                "top_1_accuracy": result.top_1_accuracy,
                "top_3_accuracy": result.top_3_accuracy,
                "top_5_accuracy": result.top_5_accuracy,
                "top_10_accuracy": result.top_10_accuracy,
                "average_time": result.average_time,
                "total_predictions": result.total_predictions,
                "valid_pathways": result.valid_pathways
            })
        
        try:
            with open("evaluation_results.json", "w") as f:
                json.dump(results_data, f, indent=2)
            
            # Save comparison table
            with open("comparison_table.json", "w") as f:
                json.dump(comparison, f, indent=2)
            
            print("Results saved to evaluation_results.json and comparison_table.json")
        except Exception as e:
            print(f"Could not save results to file: {e}")
            print("Results displayed in console only.")
    
    def plot_results(self, results: List[EvaluationResult], save_plot: bool = True):
        """Plot evaluation results."""
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = [r.method_name for r in results]
        
        # Top-k accuracy plot
        top_1 = [r.top_1_accuracy for r in results]
        top_3 = [r.top_3_accuracy for r in results]
        top_5 = [r.top_5_accuracy for r in results]
        top_10 = [r.top_10_accuracy for r in results]
        
        x = np.arange(len(methods))
        width = 0.2
        
        ax1.bar(x - 1.5*width, top_1, width, label='Top-1', alpha=0.8)
        ax1.bar(x - 0.5*width, top_3, width, label='Top-3', alpha=0.8)
        ax1.bar(x + 0.5*width, top_5, width, label='Top-5', alpha=0.8)
        ax1.bar(x + 1.5*width, top_10, width, label='Top-10', alpha=0.8)
        
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Top-k Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average time plot
        times = [r.average_time for r in results]
        ax2.bar(methods, times, alpha=0.8, color='orange')
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Average Time (s)')
        ax2.set_title('Average Prediction Time')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Valid pathways ratio
        valid_ratios = [r.valid_pathways / r.total_predictions if r.total_predictions > 0 else 0 
                       for r in results]
        ax3.bar(methods, valid_ratios, alpha=0.8, color='green')
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Valid Pathways Ratio')
        ax3.set_title('Ratio of Valid Pathways')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Accuracy vs Time scatter plot
        ax4.scatter(times, top_5, s=100, alpha=0.7)
        for i, method in enumerate(methods):
            ax4.annotate(method, (times[i], top_5[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Average Time (s)')
        ax4.set_ylabel('Top-5 Accuracy')
        ax4.set_title('Accuracy vs Speed Trade-off')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            try:
                plt.savefig("evaluation_results.png", 
                           dpi=300, bbox_inches='tight')
                print("Plot saved to evaluation_results.png")
            except Exception as e:
                print(f"Could not save plot: {e}")
        
        plt.show()

class BaselineMethod:
    """Simple baseline method for comparison."""
    
    def __init__(self, name: str = "Baseline"):
        self.name = name
        self.simple_rules = [
            {"pattern": "=O", "precursor": "OH", "prob": 0.6},
            {"pattern": "OH", "precursor": "=O", "prob": 0.5},
            {"pattern": "Cl", "precursor": "OH", "prob": 0.4},
            {"pattern": "COO", "precursor": "COOH.OH", "prob": 0.7},
        ]
    
    def predict_retrosynthesis(self, target: Molecule, num_pathways: int = 10):
        """Simple rule-based retrosynthesis prediction."""
        pathways = []
        
        for rule in self.simple_rules:
            if rule["pattern"] in target.smiles:
                # Apply rule
                precursor_smiles = target.smiles.replace(rule["pattern"], rule["precursor"])
                
                if "." in precursor_smiles:
                    # Multiple precursors
                    precursor_list = [Molecule(s) for s in precursor_smiles.split(".")]
                else:
                    precursor_list = [Molecule(precursor_smiles)]
                
                reaction = Reaction(
                    reactants=precursor_list,
                    products=[target],
                    reaction_type="simple_rule",
                    probability=rule["prob"]
                )
                
                pathways.append([reaction])
        
        return pathways[:num_pathways]

def main():
    """Run evaluation and comparison of different methods."""
    print("Bayesian Retrosynthesis Evaluation")
    print("=" * 50)
    
    # Create evaluator
    evaluator = RetrosynthesisEvaluator()
    
    # Create methods to compare
    methods = []
    
    # 1. Bayesian method with simple forward model
    from bayesian_retrosynthesis import SimpleForwardModel
    simple_forward = SimpleForwardModel()
    bayesian_simple = BayesianRetrosynthesis(
        forward_model=simple_forward,
        max_depth=3,
        num_samples=50,
        temperature=1.0
    )
    methods.append((bayesian_simple, "Bayesian (Simple)"))
    
    # 2. Bayesian method with molecular transformer
    transformer = MolecularTransformer()
    bayesian_transformer = BayesianRetrosynthesis(
        forward_model=transformer,
        max_depth=3,
        num_samples=50,
        temperature=1.0
    )
    methods.append((bayesian_transformer, "Bayesian (Transformer)"))
    
    # 3. Baseline method
    baseline = BaselineMethod()
    methods.append((baseline, "Rule-based Baseline"))
    
    # 4. Just the molecular transformer (forward only)
    methods.append((transformer, "Molecular Transformer"))
    
    # Run comparison
    comparison_results = evaluator.compare_methods(methods)
    
    # Print results
    print("\nComparison Results:")
    print("-" * 80)
    
    table = comparison_results["comparison_table"]
    
    # Print header
    print(f"{'Method':<25} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Top-10':<8} {'Time':<10} {'Valid':<15}")
    print("-" * 80)
    
    # Print rows
    for i in range(len(table["method"])):
        print(f"{table['method'][i]:<25} "
              f"{table['top_1'][i]:<8} "
              f"{table['top_3'][i]:<8} "
              f"{table['top_5'][i]:<8} "
              f"{table['top_10'][i]:<8} "
              f"{table['avg_time'][i]:<10} "
              f"{table['valid_pathways'][i]:<15}")
    
    # Plot results
    results = comparison_results["individual_results"]
    evaluator.plot_results(results)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
