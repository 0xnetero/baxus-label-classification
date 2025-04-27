#!/usr/bin/env python3
import os
import json
import time
import cv2
from tqdm import tqdm
from src.east import east_detect_text_regions, recognize_text, load_east_model
from src.tesseract import get_text_boxes
from src.eval import evaluate_ocr_accuracy, run_evaluation
import concurrent.futures
import threading

# Thread-local storage to avoid sharing resources between threads
thread_local = threading.local()

def get_model():
    """Get EAST model for the current thread"""
    if not hasattr(thread_local, "model"):
        thread_local.model = load_east_model()
    return thread_local.model

def process_image(args):
    """
    Process a single image with the specified detector method.
    
    Args:
        args: Tuple containing (image_path, image_id, detector_type, min_conf, model_path)
        
    Returns:
        tuple: (image_id, detected_text)
    """
    image_path, image_id, detector_type, min_conf, model_path = args
    
    try:
        # Use appropriate detector function
        if detector_type == "east":
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                return image_id, []
                
            # Load model (thread-safe)
            net = get_model() if model_path else load_east_model(model_path)
            
            # Detect text regions
            boxes = east_detect_text_regions(image, net, min_confidence=min_conf/100)
            
            # Recognize text in each region
            texts = []
            for box in boxes:
                text, conf = recognize_text(image, box)
                if text.strip():
                    texts.append(text)
                    
            return image_id, texts
        else:  # tesseract
            # Get bounding boxes and text using Tesseract
            _, texts, _ = get_text_boxes(image_path, min_conf)
            
            # Filter out empty strings
            texts = [text for text in texts if text.strip()]
            
            return image_id, texts
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_id, []

def process_directory_with_tqdm(directory_path, detector_type, min_conf=0, output_file=None, model_path=None, num_threads=1):
    """
    Process all images in a directory with progress bar using tqdm, optionally using multi-threading.
    
    Args:
        directory_path (str): Path to directory containing images
        detector_type (str): Type of detector to use ('east' or 'tesseract')
        min_conf (int): Minimum confidence threshold for text detection
        output_file (str, optional): Path to save results as JSON
        model_path (str, optional): Path to EAST model (only for east detector)
        num_threads (int): Number of threads to use (default: 1)
        
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
    
    print(f"Found {len(image_files)} images to process")
    
    if num_threads > 1:
        # Multi-threaded processing
        tasks = [(os.path.join(directory_path, filename), 
                 os.path.splitext(filename)[0], 
                 detector_type, 
                 min_conf, 
                 model_path) for filename in image_files]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Process images with progress bar
            futures = {executor.submit(process_image, task): task for task in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(tasks), 
                              desc="Processing images", 
                              unit="image"):
                try:
                    image_id, detected_text = future.result()
                    results[image_id] = detected_text
                except Exception as e:
                    file_path = futures[future][0]
                    tqdm.write(f"Error processing {file_path}: {e}")
                    continue
    else:
        # Single-threaded processing
        for filename in tqdm(image_files, desc="Processing images", unit="image"):
            image_path = os.path.join(directory_path, filename)
            image_id = os.path.splitext(filename)[0]
            
            if detector_type == "east":
                try:
                    # Load the image
                    image = cv2.imread(image_path)
                    if image is None:
                        tqdm.write(f"Error loading image {filename}")
                        continue
                    
                    # Load model
                    net = load_east_model(model_path)
                    
                    # Detect text regions
                    boxes = east_detect_text_regions(image, net, min_confidence=min_conf/100)
                    
                    # Recognize text in each region
                    texts = []
                    for box in boxes:
                        try:
                            text, conf = recognize_text(image, box)
                            if text.strip():
                                texts.append(text)
                        except Exception as e:
                            tqdm.write(f"Error recognizing text in region for {filename}: {e}")
                            continue
                    
                    results[image_id] = texts
                except Exception as e:
                    tqdm.write(f"Error processing {filename} with EAST: {e}")
                    # Add empty result to avoid missing keys
                    results[image_id] = []
            else:  # tesseract
                try:
                    # Get bounding boxes and text using Tesseract
                    _, texts, _ = get_text_boxes(image_path, min_conf)
                    
                    # Filter out empty strings
                    texts = [text for text in texts if text.strip()]
                    
                    results[image_id] = texts
                except Exception as e:
                    tqdm.write(f"Error processing {filename} with Tesseract: {e}")
                    # Add empty result to avoid missing keys
                    results[image_id] = []
    
    # Save results to file if specified
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")
    
    return results

def download_east_model():
    """
    Download the EAST text detection model if it doesn't exist.
    
    Returns:
        str: Path to the downloaded model
    """
    import requests
    from pathlib import Path
    
    # Create models directory if it doesn't exist
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "frozen_east_text_detection.pb"
    
    # Check if model already exists
    if model_path.exists():
        print(f"EAST model already exists at {model_path}")
        return str(model_path)
    
    # Download model
    print("Downloading EAST text detection model...")
    url = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save model
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Model downloaded successfully to {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please download the model manually from:")
        print(url)
        return None

def main():
    print("Starting OCR evaluation...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate OCR methods')
    parser.add_argument('--threads', type=int, default=1, 
                        help='Number of threads to use for processing (default: 1)')
    parser.add_argument('--tesseract-only', action='store_true',
                        help='Only evaluate the Tesseract OCR')
    parser.add_argument('--east-only', action='store_true',
                        help='Only evaluate the EAST + Tesseract OCR')
    args = parser.parse_args()
    
    # Determine if we should use verbose mode (only in single-thread mode)
    verbose = args.threads <= 1
    
    # Create test_output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Download EAST model
    east_model_path = None
    if not args.tesseract_only:
        try:
            east_model_path = download_east_model()
            if not east_model_path:
                print("EAST model not found. Only Tesseract OCR will be evaluated.")
        except Exception as e:
            print(f"Error downloading EAST model: {e}")
            print("Only Tesseract OCR will be evaluated.")
    
    # Run OCR on all images with different confidence thresholds
    conf_thresholds = [0, 20, 40, 60, 80]
    # conf_thresholds = [80]  # Use this for quicker testing

    # Store results for each method
    results = {
        "tesseract": {},
        "east": {}
    }
    
    # Print multi-threading info
    if args.threads > 1:
        print(f"\nRunning in multi-threaded mode with {args.threads} threads")
    else:
        print("\nRunning in single-threaded mode with verbose output")
    
    # Test standard Tesseract OCR
    if not args.east_only:
        print("\n===== Evaluating Standard Tesseract OCR (all images in 'images' folder) =====")
        for conf in tqdm(conf_thresholds, desc="Testing confidence thresholds", unit="level"):
            tqdm.write(f"\n=== Testing with minimum confidence: {conf} ===")
            
            try:
                # Set output filename for this confidence level
                conf_output = f"test_output/tesseract_ocr_results_conf{conf}.json"
                
                # Process images with progress bar
                start_time = time.time()
                ocr_results = process_directory_with_tqdm("images", "tesseract", 
                                                        min_conf=conf, output_file=conf_output,
                                                        num_threads=args.threads)
                ocr_time = time.time() - start_time
                
                tqdm.write(f"OCR processing completed in {ocr_time:.2f} seconds.")
                tqdm.write(f"Processed {len(ocr_results)} images.")
                
                # Evaluate results
                start_time = time.time()
                eval_output = f"test_output/tesseract_eval_results_conf{conf}.json"
                
                tqdm.write(f"Evaluating results...")
                eval_results = run_evaluation(conf_output, eval_output, min_confidence=0.5, verbose=verbose)
                eval_time = time.time() - start_time
                
                # Store results for this confidence level
                results["tesseract"][conf] = {
                    "accuracy": eval_results["accuracy"],
                    "total_samples": eval_results["total_samples"],
                    "correct_matches": eval_results["correct_matches"],
                    "ocr_time": ocr_time,
                    "eval_time": eval_time
                }
                
                tqdm.write(f"Evaluation completed in {eval_time:.2f} seconds.")
                tqdm.write(f"Accuracy: {eval_results['accuracy']:.2f}%")
                tqdm.write(f"Correct matches: {eval_results['correct_matches']}/{eval_results['total_samples']}")
            except Exception as e:
                tqdm.write(f"Error evaluating Tesseract with confidence {conf}: {e}")
                continue

    # Test EAST + Tesseract OCR
    if east_model_path and not args.tesseract_only:
        print("\n===== Evaluating EAST + Tesseract OCR (all images in 'images' folder) =====")
        for conf in tqdm(conf_thresholds, desc="Testing confidence thresholds", unit="level"):
            tqdm.write(f"\n=== Testing with minimum confidence: {conf} ===")
            
            try:
                # Set output filename for this confidence level
                conf_output = f"test_output/east_ocr_results_conf{conf}.json"
                
                # Process images with progress bar
                start_time = time.time()
                ocr_results = process_directory_with_tqdm("images", "east", 
                                                        min_conf=conf, output_file=conf_output,
                                                        model_path=east_model_path,
                                                        num_threads=args.threads)
                ocr_time = time.time() - start_time
                
                tqdm.write(f"OCR processing completed in {ocr_time:.2f} seconds.")
                tqdm.write(f"Processed {len(ocr_results)} images.")
                
                # Evaluate results
                start_time = time.time()
                eval_output = f"test_output/east_eval_results_conf{conf}.json"
                
                tqdm.write(f"Evaluating results...")
                eval_results = run_evaluation(conf_output, eval_output, min_confidence=0.5, verbose=verbose)
                eval_time = time.time() - start_time
                
                # Store results for this confidence level
                results["east"][conf] = {
                    "accuracy": eval_results["accuracy"],
                    "total_samples": eval_results["total_samples"],
                    "correct_matches": eval_results["correct_matches"],
                    "ocr_time": ocr_time,
                    "eval_time": eval_time
                }
                
                tqdm.write(f"Evaluation completed in {eval_time:.2f} seconds.")
                tqdm.write(f"Accuracy: {eval_results['accuracy']:.2f}%")
                tqdm.write(f"Correct matches: {eval_results['correct_matches']}/{eval_results['total_samples']}")
            except Exception as e:
                tqdm.write(f"Error evaluating EAST with confidence {conf}: {e}")
                continue
    
    # Save summary results
    try:
        with open("test_output/ocr_accuracy_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving summary results: {e}")
    
    # Print final summary
    if results["tesseract"]:
        print("\n=== SUMMARY: Standard Tesseract OCR ===")
        print("Confidence threshold | Accuracy | Correct/Total matches")
        print("-" * 60)
        for conf, data in results["tesseract"].items():
            print(f"{conf:20} | {data['accuracy']:8.2f}% | {data['correct_matches']}/{data['total_samples']}")
    
    if east_model_path and results["east"]:
        print("\n=== SUMMARY: EAST + Tesseract OCR ===")
        print("Confidence threshold | Accuracy | Correct/Total matches")
        print("-" * 60)
        for conf, data in results["east"].items():
            print(f"{conf:20} | {data['accuracy']:8.2f}% | {data['correct_matches']}/{data['total_samples']}")
        
        # Compare the best results from each method
        # Only try to find max if there are items in the dictionaries
        if results["tesseract"] and results["east"]:
            try:
                best_tesseract = max(results["tesseract"].items(), key=lambda x: x[1]["accuracy"])
                best_east = max(results["east"].items(), key=lambda x: x[1]["accuracy"])
                
                print("\n=== Best Results Comparison ===")
                print(f"Tesseract (conf={best_tesseract[0]}): {best_tesseract[1]['accuracy']:.2f}% accuracy")
                print(f"EAST (conf={best_east[0]}): {best_east[1]['accuracy']:.2f}% accuracy")
                
                # Calculate improvement
                improvement = best_east[1]['accuracy'] - best_tesseract[1]['accuracy']
                print(f"Improvement with EAST: {improvement:.2f}%")
            except Exception as e:
                print(f"Error comparing best results: {e}")
        elif results["east"]:
            try:
                best_east = max(results["east"].items(), key=lambda x: x[1]["accuracy"])
                print("\n=== Best EAST Results ===")
                print(f"EAST (conf={best_east[0]}): {best_east[1]['accuracy']:.2f}% accuracy")
            except Exception as e:
                print(f"Error finding best EAST result: {e}")
        elif results["tesseract"]:
            try:
                best_tesseract = max(results["tesseract"].items(), key=lambda x: x[1]["accuracy"])
                print("\n=== Best Tesseract Results ===")
                print(f"Tesseract (conf={best_tesseract[0]}): {best_tesseract[1]['accuracy']:.2f}% accuracy")
            except Exception as e:
                print(f"Error finding best Tesseract result: {e}")

if __name__ == "__main__":
    main() 