import os
import csv
import json
from pathlib import Path
from src.helpers import find_best_wine_match
from src.wine_names import WINE_NAMES

def evaluate_ocr_accuracy(ocr_results, ground_truth_file='dataset.csv', min_confidence=0.5):
    """
    Evaluate the accuracy of OCR model predictions against ground truth wine names.
    
    Args:
        ocr_results (dict): Dictionary with image_id as key and OCR detected text tokens as value.
                           Format: {image_id: ["token1", "token2", ...], ...}
        ground_truth_file (str): Path to CSV file containing ground truth wine names (default: dataset.csv)
        min_confidence (float): Minimum confidence threshold for wine name matching (default: 0.5)
    
    Returns:
        dict: Dictionary containing evaluation metrics:
            - accuracy: Overall accuracy percentage
            - total_samples: Number of samples evaluated
            - correct_matches: Number of correct matches
            - results: Detailed results for each sample
    """
    # Load ground truth data
    ground_truth = {}
    try:
        with open(ground_truth_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get('id')
                if image_id:
                    ground_truth[image_id] = row.get('name', '')
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return {
            "accuracy": 0.0,
            "total_samples": 0,
            "correct_matches": 0,
            "results": []
        }
    
    # Initialize evaluation metrics
    total_samples = 0
    correct_matches = 0
    detailed_results = []
    
    # Evaluate each OCR result
    for image_id, tokens in ocr_results.items():
        if not tokens or image_id not in ground_truth:
            continue
        
        total_samples += 1
        ground_truth_name = ground_truth[image_id]
        
        # Find best match using helpers.find_best_wine_match
        best_match, confidence = find_best_wine_match(tokens, min_score=min_confidence)
        
        # Check if match is correct (normalize strings for comparison)
        is_correct = False
        if best_match:
            # Normalize strings by removing/standardizing special characters
            normalized_match = best_match.lower().replace("'", "").replace("'", "").strip()
            normalized_truth = ground_truth_name.lower().replace("'", "").replace("'", "").strip()
            is_correct = normalized_match == normalized_truth
            
            if is_correct:
                correct_matches += 1
        
        # Store detailed result
        detailed_results.append({
            "image_id": image_id,
            "ocr_tokens": tokens,
            "ground_truth": ground_truth_name,
            "predicted_match": best_match,
            "confidence": confidence,
            "is_correct": is_correct
        })
    
    # Calculate accuracy
    accuracy = (correct_matches / total_samples * 100) if total_samples > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_matches": correct_matches,
        "results": detailed_results
    }

def run_evaluation(ocr_results_file, output_file=None, min_confidence=0.5, verbose=False):
    """
    Run evaluation on OCR results and optionally save to output file.
    
    Args:
        ocr_results_file (str): Path to JSON file containing OCR results
        output_file (str, optional): Path to save evaluation results
        min_confidence (float): Minimum confidence threshold for wine name matching
        verbose (bool): Whether to print detailed results for each image
    
    Returns:
        dict: Evaluation metrics
    """
    # Load OCR results
    ocr_results = {}
    try:
        with open(ocr_results_file, 'r') as f:
            ocr_results = json.load(f)
    except Exception as e:
        print(f"Error loading OCR results file: {e}")
        return None
    
    # Run evaluation
    evaluation_results = evaluate_ocr_accuracy(ocr_results, min_confidence=min_confidence)
    
    # Print summary
    print(f"Evaluation Results:")
    print(f"  Accuracy: {evaluation_results['accuracy']:.2f}%")
    print(f"  Correct matches: {evaluation_results['correct_matches']}/{evaluation_results['total_samples']}")
    
    # Print detailed results if verbose
    if verbose:
        print("\nDetailed Results:")
        for result in evaluation_results['results']:
            print(f"Image: {result['image_id']}")
            print(f"  OCR Tokens: {result['ocr_tokens']}")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  Predicted: {result['predicted_match']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Correct: {'✓' if result['is_correct'] else '✗'}")
            print()
    
    # Save results if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Evaluation results saved to {output_file}")
    
    return evaluation_results

if __name__ == "__main__":
    # Example usage:
    # python -m src.eval ocr_results.json evaluation_results.json
    import sys
    
    if len(sys.argv) > 1:
        ocr_results_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        min_confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        verbose = sys.argv[4] == 'True' if len(sys.argv) > 4 else False
        run_evaluation(ocr_results_file, output_file, min_confidence, verbose)
    else:
        print("Usage: python -m src.eval <ocr_results_file> [output_file] [min_confidence] [verbose]")
