#!/usr/bin/env python3
import os
import json
import time
from tqdm import tqdm
from src.tesseract import detect_text
from src.eval import evaluate_ocr_accuracy, run_evaluation

def process_directory_with_tqdm(directory_path, min_conf=0, output_file=None):
    """
    Process all images in a directory with progress bar using tqdm.
    
    Args:
        directory_path (str): Path to directory containing images
        min_conf (int): Minimum confidence threshold for text detection
        output_file (str, optional): Path to save results as JSON
        
    Returns:
        dict: Dictionary with image_id as key and list of detected text as value
    """
    import os
    import json
    from pathlib import Path
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist")
    
    # Process all image files in directory
    results = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Get list of image files
    image_files = []
    for filename in os.listdir(directory_path):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            image_files.append(filename)
    
    # Process images with progress bar
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        image_path = os.path.join(directory_path, filename)
        image_id = os.path.splitext(filename)[0]
        try:
            detected_text = detect_text(image_path, min_conf)
            results[image_id] = detected_text
            # Don't print here to keep progress bar clean
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")
    
    # Save results to file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

def main():
    print("Starting OCR evaluation...")
    
    # Create test_output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Run OCR on all images with different confidence thresholds
    conf_thresholds = [0, 20, 40, 60, 80]
    
    results = {}
    
    # Using tqdm for overall progress across confidence thresholds
    for conf in tqdm(conf_thresholds, desc="Testing confidence thresholds", unit="level"):
        tqdm.write(f"\n=== Testing with minimum confidence: {conf} ===")
        
        # Set output filename for this confidence level
        conf_output = f"test_output/ocr_results_conf{conf}.json"
        
        # Process images with progress bar
        start_time = time.time()
        ocr_results = process_directory_with_tqdm("images", min_conf=conf, output_file=conf_output)
        ocr_time = time.time() - start_time
        
        tqdm.write(f"OCR processing completed in {ocr_time:.2f} seconds.")
        tqdm.write(f"Processed {len(ocr_results)} images.")
        
        # Evaluate results
        start_time = time.time()
        eval_output = f"test_output/eval_results_conf{conf}.json"
        
        tqdm.write(f"Evaluating results...")
        eval_results = run_evaluation(conf_output, eval_output, min_confidence=0.5)
        eval_time = time.time() - start_time
        
        # Store results for this confidence level
        results[conf] = {
            "accuracy": eval_results["accuracy"],
            "total_samples": eval_results["total_samples"],
            "correct_matches": eval_results["correct_matches"],
            "ocr_time": ocr_time,
            "eval_time": eval_time
        }
        
        tqdm.write(f"Evaluation completed in {eval_time:.2f} seconds.")
        tqdm.write(f"Accuracy: {eval_results['accuracy']:.2f}%")
        tqdm.write(f"Correct matches: {eval_results['correct_matches']}/{eval_results['total_samples']}")
    
    # Save summary results
    with open("test_output/accuracy_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print("\n=== SUMMARY ===")
    print("Confidence threshold | Accuracy | Correct/Total matches")
    print("-" * 60)
    for conf, data in results.items():
        print(f"{conf:20} | {data['accuracy']:8.2f}% | {data['correct_matches']}/{data['total_samples']}")

if __name__ == "__main__":
    main() 