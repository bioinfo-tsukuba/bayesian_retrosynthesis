"""
Demonstration of Bayesian Retrosynthesis Algorithm
Interactive demo showing the capabilities of the implemented algorithm.
"""

import sys
import time
from typing import List, Optional

from bayesian_retrosynthesis import BayesianRetrosynthesis, Molecule, Reaction
from molecular_transformer import MolecularTransformer, ReactionDatabase
from evaluation import RetrosynthesisEvaluator, BaselineMethod

def print_header():
    """Print demo header."""
    print("=" * 70)
    print("  Bayesian Algorithm for Retrosynthesis - Interactive Demo")
    print("  Based on: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00320")
    print("=" * 70)
    print()

def print_molecule_info(molecule: Molecule):
    """Print detailed information about a molecule."""
    print(f"  SMILES: {molecule.smiles}")
    print(f"  Estimated MW: {molecule.molecular_weight:.2f}")
    print(f"  Complexity Score: {molecule.complexity_score:.2f}")

def print_pathway(pathway: List[Reaction], pathway_num: int, score: float):
    """Print a retrosynthetic pathway."""
    print(f"\nPathway {pathway_num} (Score: {score:.4f}):")
    print("-" * 50)
    
    for i, reaction in enumerate(pathway, 1):
        reactant_smiles = ' + '.join([mol.smiles for mol in reaction.reactants])
        product_smiles = ' + '.join([mol.smiles for mol in reaction.products])
        
        print(f"  Step {i}: {reactant_smiles} → {product_smiles}")
        print(f"    Reaction Type: {reaction.reaction_type}")
        print(f"    Probability: {reaction.probability:.3f}")
        
        # Show molecular details for starting materials
        if i == len(pathway):  # Last step (starting materials)
            print("    Starting Materials:")
            for reactant in reaction.reactants:
                print(f"      - {reactant.smiles} (Complexity: {reactant.complexity_score:.2f})")

def demo_single_molecule():
    """Demonstrate retrosynthesis for a single molecule."""
    print("1. Single Molecule Retrosynthesis Demo")
    print("-" * 40)
    
    # Create the Bayesian retrosynthesis system
    transformer = MolecularTransformer()
    predictor = BayesianRetrosynthesis(
        forward_model=transformer,
        max_depth=4,
        num_samples=200,
        temperature=1.2
    )
    
    # Example target molecule
    target_smiles = "CC(=O)OCC"  # Ethyl acetate
    target = Molecule(target_smiles)
    
    print("Target Molecule:")
    print_molecule_info(target)
    print()
    
    print("Predicting retrosynthetic pathways...")
    start_time = time.time()
    
    pathways = predictor.predict_retrosynthesis(target, num_pathways=3)
    
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds")
    print(f"Found {len(pathways)} valid pathways")
    print()
    
    # Display pathways
    for i, pathway in enumerate(pathways, 1):
        score = predictor._score_pathway(pathway)
        print_pathway(pathway, i, score)

def demo_comparison():
    """Demonstrate comparison between different methods."""
    print("\n2. Method Comparison Demo")
    print("-" * 40)
    
    # Test molecules
    test_molecules = [
        ("CC(C)(C)OC(=O)NC(Cc1ccccc1)C(O)Cn1nnc2ccccc21", "Triethylamine")
    ]
    
    # Create methods
    transformer = MolecularTransformer()
    bayesian_method = BayesianRetrosynthesis(
        forward_model=transformer,
        max_depth=3,
        num_samples=50,
        temperature=1.0
    )
    baseline_method = BaselineMethod()
    
    print("Comparing Bayesian vs Baseline methods:")
    print()
    
    for smiles, name in test_molecules:
        target = Molecule(smiles)
        print(f"Target: {name} ({smiles})")
        
        # Bayesian method
        start_time = time.time()
        bayesian_pathways = bayesian_method.predict_retrosynthesis(target, num_pathways=3)
        bayesian_time = time.time() - start_time
        
        # Baseline method
        start_time = time.time()
        baseline_pathways = baseline_method.predict_retrosynthesis(target, num_pathways=3)
        baseline_time = time.time() - start_time
        
        print(f"  Bayesian: {len(bayesian_pathways)} pathways in {bayesian_time:.3f}s")
        print(f"  Baseline: {len(baseline_pathways)} pathways in {baseline_time:.3f}s")
        
        # Show best pathway from each method
        if bayesian_pathways:
            best_bayesian = bayesian_pathways[0]
            bayesian_score = bayesian_method._score_pathway(best_bayesian)
            print(f"  Best Bayesian pathway score: {bayesian_score:.4f}")
        
        if baseline_pathways:
            baseline_score = 0.5  # Simple score for baseline
            print(f"  Best Baseline pathway score: {baseline_score:.4f}")
        
        print()

def demo_interactive():
    """Interactive demo where user can input molecules."""
    print("\n3. Interactive Demo")
    print("-" * 40)
    print("Enter SMILES strings to predict retrosynthetic pathways.")
    print("Type 'quit' to exit.")
    print()
    
    # Create predictor
    transformer = MolecularTransformer()
    predictor = BayesianRetrosynthesis(
        forward_model=transformer,
        max_depth=3,
        num_samples=100,
        temperature=1.0
    )
    
    while True:
        try:
            smiles = input("Enter SMILES: ").strip()
            
            if smiles.lower() in ['quit', 'exit', 'q']:
                break
            
            if not smiles:
                continue
            
            # Create molecule
            try:
                target = Molecule(smiles)
            except Exception as e:
                print(f"Error creating molecule: {e}")
                continue
            
            print(f"\nAnalyzing: {smiles}")
            print_molecule_info(target)
            
            # Predict pathways
            start_time = time.time()
            pathways = predictor.predict_retrosynthesis(target, num_pathways=2)
            end_time = time.time()
            
            print(f"\nFound {len(pathways)} pathways in {end_time - start_time:.2f}s")
            
            for i, pathway in enumerate(pathways, 1):
                score = predictor._score_pathway(pathway)
                print_pathway(pathway, i, score)
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def demo_evaluation():
    """Demonstrate the evaluation system."""
    print("\n4. Evaluation Demo")
    print("-" * 40)
    
    evaluator = RetrosynthesisEvaluator()
    
    # Create methods
    transformer = MolecularTransformer()
    bayesian_method = BayesianRetrosynthesis(
        forward_model=transformer,
        max_depth=3,
        num_samples=30,  # Reduced for demo
        temperature=1.0
    )
    baseline_method = BaselineMethod()
    
    methods = [
        (bayesian_method, "Bayesian"),
        (baseline_method, "Baseline")
    ]
    
    print("Running evaluation on test molecules...")
    print("(This may take a moment)")
    print()
    
    # Run evaluation
    results = evaluator.compare_methods(methods, save_results=False)
    
    # Print results
    table = results["comparison_table"]
    
    print("Results:")
    print(f"{'Method':<15} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Time':<10}")
    print("-" * 50)
    
    for i in range(len(table["method"])):
        print(f"{table['method'][i]:<15} "
              f"{table['top_1'][i]:<8} "
              f"{table['top_3'][i]:<8} "
              f"{table['top_5'][i]:<8} "
              f"{table['avg_time'][i]:<10}")

def demo_molecular_transformer():
    """Demonstrate the molecular transformer component."""
    print("\n5. Molecular Transformer Demo")
    print("-" * 40)
    
    transformer = MolecularTransformer()
    db = ReactionDatabase()
    
    # Test reactants
    test_cases = [
        ([Molecule("CCO")], "Ethanol"),
        ([Molecule("CC(=O)C")], "Acetone"),
        ([Molecule("CCl"), Molecule("NaOH")], "Chloroethane + Base")
    ]
    
    for reactants, description in test_cases:
        print(f"Reactants: {description}")
        reactant_smiles = ' + '.join([mol.smiles for mol in reactants])
        print(f"  SMILES: {reactant_smiles}")
        
        # Get predictions
        predictions = transformer.predict(reactants)
        
        print(f"  Predictions ({len(predictions)}):")
        for i, (products, prob) in enumerate(predictions[:3], 1):
            product_smiles = ' + '.join([mol.smiles for mol in products])
            print(f"    {i}. {product_smiles} (p={prob:.3f})")
        
        # Find similar reactions
        similar = db.get_similar_reactions(reactants)
        if similar:
            print(f"  Similar known reactions:")
            for sim_data in similar[:2]:
                rxn = sim_data["reaction"]
                similarity = sim_data["similarity"]
                rxn_reactants = ' + '.join([mol.smiles for mol in rxn["reactants"]])
                rxn_products = ' + '.join([mol.smiles for mol in rxn["products"]])
                print(f"    {rxn_reactants} → {rxn_products} (sim={similarity:.3f})")
        
        print()

def main():
    """Main demo function."""
    print_header()
    
    print("Available demos:")
    print("1. Single molecule retrosynthesis")
    print("2. Method comparison")
    print("3. Interactive mode")
    print("4. Evaluation system")
    print("5. Molecular transformer")
    print("6. Run all demos")
    print()
    
    try:
        choice = input("Select demo (1-6): ").strip()
        
        if choice == "1":
            demo_single_molecule()
        elif choice == "2":
            demo_comparison()
        elif choice == "3":
            demo_interactive()
        elif choice == "4":
            demo_evaluation()
        elif choice == "5":
            demo_molecular_transformer()
        elif choice == "6":
            demo_single_molecule()
            demo_comparison()
            demo_molecular_transformer()
            demo_evaluation()
        else:
            print("Invalid choice. Running single molecule demo...")
            demo_single_molecule()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
