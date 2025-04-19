from src.tesseract import detect_text, process_image_file, process_directory
import os
import json
import argparse

def test_single_image(image_path, min_conf=0):
    """Test the text detection on a single image"""
    print(f"Testing text detection on {image_path} with min_conf={min_conf}")
    
    # Detect text in the image
    detected_text = detect_text(image_path, min_conf)
    print(f"Detected text: {detected_text}")
    
    # Process image to get results in the format for evaluation
    result = process_image_file(image_path, min_conf)
    
    # Output as JSON for debugging
    print(json.dumps(result, indent=2))
    
    return result

def test_directory(directory_path, min_conf=0, output_file="ocr_results.json"):
    """Test the text detection on a directory of images"""
    print(f"Processing directory {directory_path} with min_conf={min_conf}")
    
    # Process all images in the directory
    results = process_directory(directory_path, min_conf, output_file)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Processed {len(results)} images")
    print(f"  Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test OCR text detection")
    parser.add_argument("-i", "--image", help="Path to single image")
    parser.add_argument("-d", "--dir", help="Path to directory of images")
    parser.add_argument("-c", "--min-conf", type=int, default=0,
                        help="Minimum confidence threshold")
    parser.add_argument("-o", "--output", default="ocr_results.json",
                        help="Output file for results (when processing directory)")
    args = parser.parse_args()
    
    # Check if running with a single image or directory
    if args.image:
        # Check if file exists
        if not os.path.isfile(args.image):
            print(f"Error: File {args.image} does not exist")
            exit(1)
        
        # Test on single image
        test_single_image(args.image, args.min_conf)
    
    elif args.dir:
        # Check if directory exists
        if not os.path.isdir(args.dir):
            print(f"Error: Directory {args.dir} does not exist")
            exit(1)
        
        # Test on directory
        test_directory(args.dir, args.min_conf, args.output)
    
    else:
        print("Please specify either an image (-i) or directory (-d)")
        
        # Default behavior: search for images folder and process the first image found
        if os.path.isdir("images"):
            print("Found 'images' directory, searching for first image...")
            for filename in os.listdir("images"):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join("images", filename)
                    print(f"Found {image_path}, running test...")
                    test_single_image(image_path, args.min_conf)
                    break
            else:
                print("No images found in 'images' directory") 