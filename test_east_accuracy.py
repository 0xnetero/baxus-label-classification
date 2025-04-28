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
        
        # Use appropriate detector function
        if detector_type == "east":
            # Load model (thread-safe)
            net = get_model() if model_path else load_east_model(model_path)
            
            # Detect text regions
            boxes = east_detect_text_regions(img, net, min_confidence=min_conf/100)
            
            # Recognize text in each region
            texts = []
            for box in boxes:
                text, conf = recognize_text(img, box)
                if text.strip():
                    texts.append(text)
                    
            return image_id, texts
        else:  # tesseract
            try:
                # Use shorter timeout for large images
                timeout_seconds = 10 if max(height, width) > 1000 else 15
                
                # Use timeout to prevent hanging on problematic images
                with time_limit(timeout_seconds):
                    # Get bounding boxes and text using Tesseract
                    _, texts, _ = get_text_boxes(image_path, min_conf)
                    
                    # Filter out empty strings
                    texts = [text for text in texts if text.strip()]
                    
                    return image_id, texts
            except TimeoutError:
                print(f"⚠️ Tesseract processing timed out for {image_path} ({timeout_seconds}s)")
                # Mark as problematic for future runs
                mark_problematic(os.path.basename(image_path))
                return image_id, []
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_id, []

def process_directory_with_tqdm(directory_path, detector_type, min_conf=0, output_file=None, model_path=None, num_threads=1, max_images=None, batch_size=50):
    """
    Process all images in a directory with progress bar using tqdm, optionally using multi-threading.
    Uses batch processing to handle large datasets efficiently.
    
    Args:
        directory_path (str): Path to directory containing images
        detector_type (str): Type of detector to use ('east' or 'tesseract')
        min_conf (int): Minimum confidence threshold for text detection
        output_file (str, optional): Path to save results as JSON
        model_path (str, optional): Path to EAST model (only for east detector)
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
    problematic_images_file = "test_output/problematic_images.txt"
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
    checkpoint_file = f"test_output/checkpoint_{detector_type}_conf{min_conf}.json"
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
                         detector_type, 
                         min_conf, 
                         model_path))
            
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
                        # Mark as problematic
                        mark_problematic(filename)
                else:  # tesseract
                    try:
                        # Use timeout to prevent hanging on problematic images
                        with time_limit(15):  # 15 second timeout
                            # Get bounding boxes and text using Tesseract
                            _, texts, _ = get_text_boxes(image_path, min_conf)
                            
                            # Filter out empty strings
                            texts = [text for text in texts if text.strip()]
                            
                            results[image_id] = texts
                    except TimeoutError:
                        tqdm.write(f"⚠️ Tesseract processing timed out for {filename} (15s)")
                        results[image_id] = []
                        # Mark as problematic for future runs
                        mark_problematic(filename)
                    except Exception as e:
                        tqdm.write(f"Error processing {filename} with Tesseract: {e}")
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
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    parser.add_argument('--conf-thresholds', type=str, default='0,20,40,60,80',
                        help='Comma-separated list of confidence thresholds to test')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of images to process in each batch (default: 50)')
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
    
    # Parse confidence thresholds
    try:
        conf_thresholds = [int(t) for t in args.conf_thresholds.split(',')]
    except:
        print(f"Invalid confidence thresholds: {args.conf_thresholds}. Using defaults.")
        conf_thresholds = [0, 20, 40, 60, 80]

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
    
    if args.max_images:
        print(f"Testing with a maximum of {args.max_images} images")
    
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
                                                        num_threads=args.threads,
                                                        max_images=args.max_images,
                                                        batch_size=args.batch_size)
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
                                                        num_threads=args.threads,
                                                        max_images=args.max_images,
                                                        batch_size=args.batch_size)
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