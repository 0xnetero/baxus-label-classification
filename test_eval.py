import json
from src.eval import evaluate_ocr_accuracy, run_evaluation

def create_sample_ocr_results():
    """Create sample OCR results for testing the evaluation function."""
    sample_results = {
        # Correct match example (Blanton's)
        "164": ["Blanton's", "Original", "Single", "Barrel"],
        
        # Misspelled example (should still match correctly)
        "2848": ["Eagle", "Rare", "10", "Yr"],
        
        # Missing key words (might not match correctly)
        "4984": ["Taylor", "Batch"],
        
        # Completely wrong tokens (shouldn't match)
        "466": ["Some", "Random", "Text"],
        
        # Barrell Vantage misspelled example
        "15984": ["Barrel", "Vantag"],
        
        # Wild Turkey example
        "4708": ["Wild", "Turkey", "101", "Proof"]
    }
    
    # Save to JSON file for testing
    with open("sample_ocr_results.json", "w") as f:
        json.dump(sample_results, f, indent=2)
    
    return sample_results

def test_evaluation():
    """Test the evaluation function with sample OCR results."""
    print("Creating sample OCR results...")
    ocr_results = create_sample_ocr_results()
    
    print("\nRunning evaluation with default parameters...")
    evaluation = evaluate_ocr_accuracy(ocr_results)
    
    print(f"\nEvaluation Results:")
    print(f"  Accuracy: {evaluation['accuracy']:.2f}%")
    print(f"  Correct matches: {evaluation['correct_matches']}/{evaluation['total_samples']}")
    
    print("\nDetailed Results:")
    for result in evaluation['results']:
        print(f"  Image ID: {result['image_id']}")
        print(f"    OCR Tokens: {result['ocr_tokens']}")
        print(f"    Ground Truth: {result['ground_truth']}")
        print(f"    Prediction: {result['predicted_match'] or 'No match'}")
        print(f"    Confidence: {result['confidence']:.4f}")
        print(f"    Correct: {'✓' if result['is_correct'] else '✗'}")
        print()
    
    # Test with different confidence threshold
    print("\nRunning evaluation with higher confidence threshold (0.7)...")
    evaluation_high_conf = evaluate_ocr_accuracy(ocr_results, min_confidence=0.7)
    print(f"  Accuracy: {evaluation_high_conf['accuracy']:.2f}%")
    print(f"  Correct matches: {evaluation_high_conf['correct_matches']}/{evaluation_high_conf['total_samples']}")
    
    # Save sample results to file
    with open("sample_evaluation_results.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    print("\nSample evaluation results saved to sample_evaluation_results.json")

if __name__ == "__main__":
    test_evaluation() 