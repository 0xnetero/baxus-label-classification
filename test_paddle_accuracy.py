#!/usr/bin/env python3
import os
import json
import time
import cv2
from tqdm import tqdm
from src.paddle_ocr import detect_text, get_text_boxes
from src.eval import evaluate_ocr_accuracy
from src.helpers import find_best_wine_match_fuzzy
import concurrent.futures
import threading
import signal
from contextlib import contextmanager
import functools

# Thread-local storage to avoid sharing resources between threads
thread_local = threading.local()

class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """
    Context manager that raises a TimeoutError if execution takes longer than specified seconds
    """
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def timeout(seconds):
    """
    Decorator to timeout a function after specified seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with time_limit(seconds):
                    return func(*args, **kwargs)
            except TimeoutError:
                return None
        return wrapper
    return decorator

def process_image(args):
    """
    Process a single image with PaddleOCR.
    
    Args:
        args: Tuple containing (image_path, image_id, min_conf)
        
    Returns:
        tuple: (image_id, detected_text)
    """
    image_path, image_id, min_conf = args
    
    try:
        # Check image size before processing
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Error loading image: {image_path}")
            return image_id, []
            
        height, width = img.shape[:2]
        
        # Skip very small images
        if height < 50 or width < 50:
            print(f"⚠️ Skipping small image {image_path} ({width}x{height})")
            return image_id, []
            
        # Resize large images
        max_dimension = 2000  # Maximum dimension for processing
        if height > max_dimension or width > max_dimension:
            print(f"⚠️ Resizing large image {image_path} ({width}x{height})")
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        try:
            # Use timeout to prevent hanging on problematic images
            timeout_seconds = 20 if max(height, width) > 1000 else 30
            
            with time_limit(timeout_seconds):
                # Detect text using PaddleOCR
                detected_text = detect_text(image_path, min_conf)
                
                # Filter out empty strings
                detected_text = [text for text in detected_text if text.strip()]
                
                return image_id, detected_text
        except TimeoutError:
            print(f"⚠️ PaddleOCR processing timed out for {image_path} ({timeout_seconds}s)")
            # Mark as problematic for future runs
            mark_problematic(os.path.basename(image_path))
            return image_id, []
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_id, []

def process_directory_with_tqdm(directory_path, min_conf=0, output_file=None, num_threads=1, max_images=None, batch_size=50):
    """
    Process all images in a directory with progress bar using tqdm, optionally using multi-threading.
    Uses batch processing to handle large datasets efficiently.
    
    Args:
        directory_path (str): Path to directory containing images
        min_conf (float): Minimum confidence threshold for text detection
        output_file (str, optional): Path to save results as JSON
        num_threads (int): Number of threads to use (default: 1)
        max_images (int, optional): Maximum number of images to process (for testing)
        batch_size (int): Number of images to process in each batch (default: 50)
        
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
    
    # Get list of image files and sort them
    image_files = []
    for filename in os.listdir(directory_path):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            image_files.append(filename)
    
    # Sort files to ensure consistent processing order
    image_files.sort()
    
    # Limit number of images if specified (for testing)
    if max_images and len(image_files) > max_images:
        print(f"Limiting to {max_images} images for testing")
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to process")
    print(f"First few images: {image_files[:5]}")
    print(f"Last few images: {image_files[-5:]}")
    
    # Load problematic images list if it exists
    problematic_images_file = "test_output/problematic_images_paddle.txt"
    problematic_images = set()
    if os.path.exists(problematic_images_file):
        with open(problematic_images_file, 'r') as f:
            problematic_images = set(line.strip() for line in f.readlines())
        if problematic_images:
            print(f"Loaded {len(problematic_images)} known problematic images to skip")
    
    # Function to add a problematic image to the list
    def mark_problematic(image_name):
        problematic_images.add(image_name)
        try:
            with open(problematic_images_file, 'a+') as f:
                f.write(f"{image_name}\n")
        except:
            pass
    
    # Load checkpoint if it exists
    checkpoint_file = f"test_output/scaled_1_5x_checkpoint_paddle_conf{min_conf}.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', {})
                processed_files = set(results.keys())
                print(f"Loaded checkpoint with {len(processed_files)} processed images")
                print(f"Last processed image: {list(processed_files)[-1] if processed_files else 'None'}")
                # Filter out already processed images
                image_files = [f for f in image_files if os.path.splitext(f)[0] not in processed_files]
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Process images in batches
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} images)")
        print(f"Batch files: {batch_files}")
        
        if num_threads > 1:
            # Multi-threaded processing for the batch
            tasks = []
            for filename in batch_files:
                # Skip known problematic images
                if filename in problematic_images:
                    print(f"⚠️ Skipping known problematic image: {filename}")
                    # Add empty result to avoid missing keys
                    results[os.path.splitext(filename)[0]] = []
                    continue
                    
                tasks.append((os.path.join(directory_path, filename), 
                         os.path.splitext(filename)[0], 
                         min_conf))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Process images with progress bar
                futures = {executor.submit(process_image, task): task for task in tasks}
                
                completed = 0
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                  total=len(tasks), 
                                  desc=f"Batch {batch_idx + 1}/{total_batches}", 
                                  unit="image"):
                    try:
                        image_id, detected_text = future.result()
                        results[image_id] = detected_text
                        
                        completed += 1
                        if completed % 10 == 0:  # More frequent updates
                            tqdm.write(f"✓ Progress: {completed}/{len(tasks)} images in batch")
                            tqdm.write(f"Current image: {image_id}")
                            
                    except Exception as e:
                        file_path = futures[future][0]
                        filename = os.path.basename(file_path)
                        tqdm.write(f"Error processing {file_path}: {e}")
                        # Mark as problematic for future runs
                        mark_problematic(filename)
                        continue
        else:
            # Single-threaded processing for the batch
            for i, filename in enumerate(tqdm(batch_files, desc=f"Batch {batch_idx + 1}/{total_batches}", unit="image")):
                # Skip known problematic images
                if filename in problematic_images:
                    tqdm.write(f"⚠️ Skipping known problematic image: {filename}")
                    # Add empty result to avoid missing keys
                    results[os.path.splitext(filename)[0]] = []
                    continue
                    
                image_path = os.path.join(directory_path, filename)
                image_id = os.path.splitext(filename)[0]
                
                tqdm.write(f"Processing image {i+1}/{len(batch_files)}: {filename}")
                
                try:
                    # Use timeout to prevent hanging on problematic images
                    with time_limit(30):  # 30 second timeout
                        # Get text using PaddleOCR
                        detected_text = detect_text(image_path, min_conf)
                        
                        # Filter out empty strings
                        detected_text = [text for text in detected_text if text.strip()]
                        
                        results[image_id] = detected_text
                except TimeoutError:
                    tqdm.write(f"⚠️ PaddleOCR processing timed out for {filename} (30s)")
                    results[image_id] = []
                    # Mark as problematic for future runs
                    mark_problematic(filename)
                except Exception as e:
                    tqdm.write(f"Error processing {filename} with PaddleOCR: {e}")
                    # Add empty result to avoid missing keys
                    results[image_id] = []
                    # Mark as problematic
                    mark_problematic(filename)
                
                # Save checkpoint after each image in single-threaded mode
                if not num_threads > 1:
                    try:
                        checkpoint_data = {'results': results}
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint_data, f, indent=2)
                        tqdm.write(f"✓ Checkpoint saved for {filename}")
                    except Exception as e:
                        tqdm.write(f"Error saving checkpoint: {e}")
        
        # Save checkpoint after each batch
        try:
            checkpoint_data = {'results': results}
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"✓ Checkpoint saved after batch {batch_idx + 1}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    # Save final results to file if specified
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")
    
    return results

# Custom evaluation function that uses the fuzzy matcher
def run_evaluation_fuzzy(ocr_results_file, output_file=None, min_confidence=0.6, verbose=False):
    """
    Run evaluation on OCR results using the fuzzy matching algorithm and optionally save to output file.
    
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
    
    # Load ground truth data
    ground_truth = {}
    ground_truth_file = 'dataset.csv'
    try:
        import csv
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
    
    # Get total number of images in the images folder
    total_images = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    for filename in os.listdir('images'):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            total_images += 1
    
    # Initialize evaluation metrics
    total_samples = total_images  # Use total number of images as denominator
    correct_matches = 0
    detailed_results = []
    
    # Evaluate each OCR result using fuzzy matching
    for image_id, tokens in ocr_results.items():
        ground_truth_name = ground_truth.get(image_id, '')
        
        # Find best match using fuzzy matching
        best_match, confidence = find_best_wine_match_fuzzy(tokens, min_score=min_confidence*100)
        
        # Check if match is correct (normalize strings for comparison)
        is_correct = False
        if best_match and ground_truth_name:
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
    
    # Calculate accuracy based on total number of images
    accuracy = (correct_matches / total_samples * 100) if total_samples > 0 else 0.0
    
    # Create final evaluation results
    evaluation_results = {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_matches": correct_matches,
        "results": detailed_results
    }
    
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
            print(f"  Predicted: {result['predicted_match'] or 'None'}")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Correct: {'✓' if result['is_correct'] else '✗'}")
            print()
    
    # Save results if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Evaluation results saved to {output_file}")
    
    return evaluation_results

def main():
    print("Starting PaddleOCR evaluation with fuzzy matching...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate PaddleOCR accuracy with fuzzy matching')
    parser.add_argument('--threads', type=int, default=1, 
                        help='Number of threads to use for processing (default: 1)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    parser.add_argument('--conf-thresholds', type=str, default='0,0.2,0.4,0.6,0.8',
                        help='Comma-separated list of confidence thresholds to test')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of images to process in each batch (default: 50)')
    parser.add_argument('--min-match-confidence', type=float, default=0.5,
                        help='Minimum confidence for fuzzy matching (default: 0.5)')
    args = parser.parse_args()
    
    # Determine if we should use verbose mode (only in single-thread mode)
    verbose = args.threads <= 1
    
    # Create test_output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Parse confidence thresholds
    try:
        conf_thresholds = [float(t) for t in args.conf_thresholds.split(',')]
    except:
        print(f"Invalid confidence thresholds: {args.conf_thresholds}. Using defaults.")
        conf_thresholds = [0, 0.2, 0.4, 0.6, 0.8]

    # Store results
    results = {}
    
    # Print multi-threading info
    if args.threads > 1:
        print(f"\nRunning in multi-threaded mode with {args.threads} threads")
    else:
        print("\nRunning in single-threaded mode with verbose output")
    
    if args.max_images:
        print(f"Testing with a maximum of {args.max_images} images")
    
    # Test PaddleOCR
    print("\n===== Evaluating PaddleOCR with fuzzy matching (all images in 'images' folder) =====")
    for conf in tqdm(conf_thresholds, desc="Testing confidence thresholds", unit="level"):
        tqdm.write(f"\n=== Testing with minimum confidence: {conf} ===")
        
        try:
            # Set output filename for this confidence level
            conf_output = f"test_output/scaled_1_5x_paddle_fuzzy_results_conf{conf}.json"
            
            # Process images with progress bar
            start_time = time.time()
            ocr_results = process_directory_with_tqdm("images", 
                                                    min_conf=conf, output_file=conf_output,
                                                    num_threads=args.threads,
                                                    max_images=args.max_images,
                                                    batch_size=args.batch_size)
            ocr_time = time.time() - start_time
            
            tqdm.write(f"OCR processing completed in {ocr_time:.2f} seconds.")
            tqdm.write(f"Processed {len(ocr_results)} images.")
            
            # Evaluate results
            start_time = time.time()
            eval_output = f"test_output/scaled_1_5x_paddle_fuzzy_eval_results_conf{conf}.json"
            
            tqdm.write(f"Evaluating results with fuzzy matching...")
            eval_results = run_evaluation_fuzzy(conf_output, eval_output, 
                                              min_confidence=args.min_match_confidence, 
                                              verbose=verbose)
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
        except Exception as e:
            tqdm.write(f"Error evaluating PaddleOCR with confidence {conf}: {e}")
            continue
    
    # Save summary results
    try:
        with open("test_output/scaled_1_5x_paddle_fuzzy_accuracy.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving summary results: {e}")
    
    # Print final summary
    print("\n=== SUMMARY: PaddleOCR with Fuzzy Matching ===")
    print("Confidence threshold | Accuracy | Correct/Total matches")
    print("-" * 60)
    for conf, data in results.items():
        print(f"{conf:20} | {data['accuracy']:8.2f}% | {data['correct_matches']}/{data['total_samples']}")
    
    if results:
        try:
            best_result = max(results.items(), key=lambda x: x[1]["accuracy"])
            print(f"\n=== Best PaddleOCR Result with Fuzzy Matching ===")
            print(f"PaddleOCR (conf={best_result[0]}): {best_result[1]['accuracy']:.2f}% accuracy")
        except Exception as e:
            print(f"Error finding best result: {e}")

if __name__ == "__main__":
    main()
