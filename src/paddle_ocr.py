# import the necessary packages
from paddleocr import PaddleOCR, draw_ocr
import cv2
import os
import json
import numpy as np

# Initialize PaddleOCR with server models for better accuracy
# The server models are larger and slower but more accurate than the mobile models
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    # Specify server models for detection and recognition
    det_model_dir=None,  # Will download the server detection model if not specified
    rec_model_dir=None,  # Will download the server recognition model if not specified
    cls_model_dir=None,  # Will download the server classification model if not specified
    use_space_char=True,  # Better handling of spaces
    use_gpu=True,        # Use GPU if available
    det_db_score_mode="slow", # Use the "slow" scoring mode for better detection accuracy
    # Use server version models instead of mobile version
    det_db_unclip_ratio=2.0, # Higher value leads to larger detection boxes
    det_limit_side_len=2560, # Higher resolution processing
    det_limit_type='max',
    rec_batch_num=6,     # Larger batch size for recognition
    rec_char_dict_path=None, # Use default dictionary
    rec_img_h=48,        # Server model uses higher resolution
)

def detect_text(image_path, min_conf=0):
    """
    Detect and localize text in an image using PaddleOCR.
    
    Args:
        image_path (str): Path to the input image
        min_conf (float): Minimum confidence threshold for text detection (default: 0)
        
    Returns:
        list: List of detected text tokens with confidence above the threshold
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise ValueError(f"Could not read image from {image_path}")
    
    # Run OCR on the image
    result = ocr.ocr(image_path, cls=True)
    
    # Extract detected text with confidence above threshold
    detected_text = []
    
    # Process the results
    if result and result[0]:
        for line in result[0]:
            # Each line has format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [text, confidence]]
            if len(line) >= 2:
                text = line[1][0]  # Text content
                conf = float(line[1][1])  # Confidence value
                
                # Add text to the list if confidence is above the threshold
                # and text is not empty
                if conf > min_conf and text.strip():
                    detected_text.append(text)
    
    return detected_text

def process_image_file(image_path, min_conf=0):
    """
    Process an image file and return detected text as a dictionary
    compatible with the evaluation function.
    
    Args:
        image_path (str): Path to the input image
        min_conf (float): Minimum confidence threshold for text detection
        
    Returns:
        dict: Dictionary with image_id as key and list of detected text as value
    """
    # Extract image ID from the filename (assuming filename is the ID with extension)
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
        min_conf (float): Minimum confidence threshold for text detection
        output_file (str, optional): Path to save results as JSON
        
    Returns:
        dict: Dictionary with image_id as key and list of detected text as value
    """
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
    Get text bounding boxes, text content, and confidence values from an image using PaddleOCR.
    
    Args:
        image_path (str): Path to the input image
        min_conf (float): Minimum confidence threshold for text detection (default: 0)
        
    Returns:
        tuple: (boxes, texts, confidences) where:
            - boxes: List of bounding boxes in format [x, y, w, h]
            - texts: List of text strings for each box
            - confidences: List of confidence values for each box
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise ValueError(f"Could not read image from {image_path}")
    
    # Load the image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Get original dimensions
    original_h, original_w = image.shape[:2]
    
    # Run OCR on the image
    result = ocr.ocr(image_path, cls=True)
    
    # Initialize lists for boxes, texts, and confidences
    boxes = []
    texts = []
    confidences = []
    
    # Process the results
    if result and result[0]:
        for line in result[0]:
            # Each line has format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [text, confidence]]
            if len(line) >= 2:
                text = line[1][0]  # Text content
                conf = float(line[1][1])  # Confidence value
                
                # Only consider results if they have text and meet minimum confidence
                if conf > min_conf and text.strip():
                    # Extract bounding box coordinates
                    # PaddleOCR returns points in quadrilateral format [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    # We need to convert to [x, y, w, h] format
                    points = line[0]
                    
                    # Find the min and max x, y coordinates to create bounding box
                    x_coords = [point[0] for point in points]
                    y_coords = [point[1] for point in points]
                    
                    x = max(0, min(x_coords))
                    y = max(0, min(y_coords))
                    max_x = min(original_w - 1, max(x_coords))
                    max_y = min(original_h - 1, max(y_coords))
                    
                    w = max_x - x
                    h = max_y - y
                    
                    # Skip boxes with zero width or height
                    if w <= 0 or h <= 0:
                        continue
                    
                    boxes.append([int(x), int(y), int(w), int(h)])
                    texts.append(text)
                    confidences.append(conf)
    
    return boxes, texts, confidences

if __name__ == "__main__":
    import argparse
    
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image to localize")
    ap.add_argument("-d", "--dir", help="path to directory of images to process")
    ap.add_argument("-c", "--min-conf", type=float, default=0,
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
            os.makedirs(os.path.dirname(args["output"]), exist_ok=True)
            with open(args["output"], 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args['output']}")
            
    elif args["dir"]:
        # Process directory of images
        process_directory(args["dir"], args["min_conf"], args["output"])
    else:
        print("Please specify either an image file (-i) or directory (-d)") 