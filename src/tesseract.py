# import the necessary packages
from pytesseract import Output
import pytesseract
import cv2

def detect_text(image_path, min_conf=0):
    """
    Detect and localize text in an image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the input image
        min_conf (int): Minimum confidence threshold for text detection (default: 0)
        
    Returns:
        list: List of detected text tokens with confidence above the threshold
    """
    # Load the input image and convert it from BGR to RGB channel ordering
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use Tesseract to localize each area of text in the input image
    results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
    # Extract detected text with confidence above threshold
    detected_text = []
    
    # Loop over each of the individual text localizations
    for i in range(0, len(results["text"])):
        # Extract the OCR text along with the confidence
        text = results["text"][i]
        conf = int(results["conf"][i])
        
        # Add text to the list if confidence is above the threshold
        # and text is not empty
        if conf > min_conf and text.strip():
            # Clean up the text (remove non-ASCII characters)
            detected_text.append(text)
    
    return detected_text

def process_image_file(image_path, min_conf=0):
    """
    Process an image file and return detected text as a dictionary
    compatible with the evaluation function.
    
    Args:
        image_path (str): Path to the input image
        min_conf (int): Minimum confidence threshold for text detection
        
    Returns:
        dict: Dictionary with image_id as key and list of detected text as value
    """
    # Extract image ID from the filename (assuming filename is the ID with extension)
    import os
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Detect text in the image
    detected_text = detect_text(image_path, min_conf)
    
    # Return dictionary with image_id as key and detected text as value
    return {image_id: detected_text}

def process_directory(directory_path, min_conf=0, output_file=None):
    """
    Process all images in a directory and return detected text for all images.
    
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
    
    for filename in os.listdir(directory_path):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            image_path = os.path.join(directory_path, filename)
            image_id = os.path.splitext(filename)[0]
            try:
                detected_text = detect_text(image_path, min_conf)
                results[image_id] = detected_text
                print(f"Processed {filename}: {detected_text}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Save results to file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

def get_text_boxes(image_path, min_conf=0):
    """
    Get text bounding boxes, text content, and confidence values from an image.
    
    Args:
        image_path (str): Path to the input image
        min_conf (int): Minimum confidence threshold for text detection (default: 0)
        
    Returns:
        tuple: (boxes, texts, confidences) where:
            - boxes: List of bounding boxes in format [x, y, w, h]
            - texts: List of text strings for each box
            - confidences: List of confidence values for each box
    """
    # Load the input image and convert it from BGR to RGB channel ordering
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Get original dimensions
    original_h, original_w = image.shape[:2]
    
    # Skip very large images
    if original_w > 2000 or original_h > 2000:
        # Resize large images
        scale = 2000 / max(original_w, original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        original_w, original_h = new_w, new_h
    
    # Convert to RGB for Tesseract
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        # Use Tesseract with a single configuration for speed
        results = pytesseract.image_to_data(rgb, config="--psm 6 --oem 1", output_type=Output.DICT)
        
        # Initialize lists for boxes, texts, and confidences
        boxes = []
        texts = []
        confidences = []
        
        # Process results
        for i in range(0, len(results["text"])):
            # Extract the OCR text along with the confidence and bounding box
            text = results["text"][i]
            conf = int(results["conf"][i])
            
            # Only consider results if they have text and meet minimum confidence
            if conf > min_conf and text.strip():
                # Extract bounding box coordinates
                x = int(results["left"][i])
                y = int(results["top"][i])
                w = int(results["width"][i])
                h = int(results["height"][i])
                
                # Skip boxes with zero width or height
                if w <= 0 or h <= 0:
                    continue
                
                # Ensure the box is not outside the image boundaries
                x = max(0, min(x, original_w - 1))
                y = max(0, min(y, original_h - 1))
                w = min(w, original_w - x)
                h = min(h, original_h - y)
                
                # Only add valid boxes
                if w > 0 and h > 0:
                    boxes.append([x, y, w, h])
                    texts.append(text)
                    confidences.append(conf / 100.0)  # Normalize to [0, 1]
        
        return boxes, texts, confidences
        
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return [], [], []

if __name__ == "__main__":
    import argparse
    
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image to localize")
    ap.add_argument("-d", "--dir", help="path to directory of images to process")
    ap.add_argument("-c", "--min-conf", type=int, default=0,
        help="minimum confidence value to filter weak text detection")
    ap.add_argument("-o", "--output", help="path to save output JSON file")
    args = vars(ap.parse_args())
    
    # Check if image or directory is specified
    if args["image"]:
        # Process single image
        result = process_image_file(args["image"], args["min_conf"])
        for image_id, text in result.items():
            print(f"Image ID: {image_id}")
            print(f"Detected Text: {text}")
        
        # Save results to file if specified
        if args["output"]:
            import json
            import os
            os.makedirs(os.path.dirname(args["output"]), exist_ok=True)
            with open(args["output"], 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args['output']}")
            
    elif args["dir"]:
        # Process directory of images
        process_directory(args["dir"], args["min_conf"], args["output"])
    else:
        print("Please specify either an image file (-i) or directory (-d)")