import cv2
import numpy as np
import pytesseract
import os
import json
from pytesseract import Output
from imutils.object_detection import non_max_suppression
from src.helpers import cleanup_text

def load_east_model(model_path=None):
    """
    Load the EAST text detection model.
    
    Args:
        model_path (str, optional): Path to the pre-trained EAST model
        
    Returns:
        net: The loaded deep learning model
    """
    # If model path is not provided, use the default model in the models directory
    if not model_path:
        model_path = os.path.join("models", "frozen_east_text_detection.pb")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"EAST model not found at {model_path}. Please download it from: "
                              f"https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb")
    
    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet(model_path)
    
    return net

def east_detect_text_regions(image, net, min_confidence=0.5, width=320, height=320):
    """
    Detect text regions in an image using the EAST text detector.
    
    Args:
        image: Input image
        net: Pre-trained EAST model
        min_confidence (float): Minimum confidence threshold for text detection
        width (int): Width of the resized image (multiple of 32)
        height (int): Height of the resized image (multiple of 32)
        
    Returns:
        list: List of bounding boxes containing [x, y, w, h]
    """
    # Get image dimensions
    orig_h, orig_w = image.shape[:2]
    
    # Set width and height to be multiples of 32
    width = 32 * (width // 32)
    height = 32 * (height // 32)
    
    # Use larger image for better detection if the original image is large
    if orig_w > 800 or orig_h > 800:
        width = 640
        height = 640
    
    # Calculate ratios for scaling
    r_w = orig_w / float(width)
    r_h = orig_h / float(height)
    
    # Resize the image to the required dimensions
    resized = cv2.resize(image, (width, height))
    
    # Get dimensions of the resized image
    (H, W) = resized.shape[:2]
    
    # Define output layer names for the EAST detector
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",  # Confidence score
        "feature_fusion/concat_3"         # Coordinates
    ]
    
    # Construct a blob from the resized image
    blob = cv2.dnn.blobFromImage(resized, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    # Forward pass through the network
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    # Get dimensions of the scores
    (num_rows, num_cols) = scores.shape[2:4]
    
    # Initialize lists to store bounding boxes and their confidence scores
    boxes = []
    confidences = []
    
    # Loop over the rows and columns of the scores
    for y in range(0, num_rows):
        # Extract the scores, coordinates, and orientation
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        
        # Loop over columns
        for x in range(0, num_cols):
            # Check if the confidence is high enough
            score = scores_data[x]
            if score < min_confidence:
                continue
            
            # Compute the offset factor
            (offset_x, offset_y) = (x * 4.0, y * 4.0)
            
            # Calculate the angle
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            # Calculate the size of the bounding box
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]
            
            # Calculate the ending coordinates of the bounding box
            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            
            # Calculate the starting coordinates of the bounding box
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            
            # Add the bounding box and confidence to their respective lists
            boxes.append((start_x, start_y, end_x, end_y))
            confidences.append(score)
    
    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    # Lower overlap threshold to merge nearby text regions
    boxes = non_max_suppression(np.array(boxes), probs=confidences, overlapThresh=0.2)
    
    # Convert boxes back to original image dimensions
    final_boxes = []
    for (start_x, start_y, end_x, end_y) in boxes:
        # Scale bounding box coordinates based on the ratios
        start_x = int(start_x * r_w)
        start_y = int(start_y * r_h)
        end_x = int(end_x * r_w)
        end_y = int(end_y * r_h)
        
        # Calculate width and height
        w = end_x - start_x
        h = end_y - start_y
        
        # Only keep boxes with reasonable dimensions
        if w > 10 and h > 5:
            # Store as [x, y, w, h] for consistency
            final_boxes.append((start_x, start_y, w, h))
    
    return final_boxes

def preprocess_for_ocr(image):
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image: Input image
        
    Returns:
        processed_image: Image processed for better OCR
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive thresholding
    # This helps with varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Noise removal
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Dilation to make text more visible
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(opening, kernel, iterations=1)
    
    return processed_image

def recognize_text(image, box, padding=0.1):
    """
    Recognize text within a bounding box using Tesseract OCR.
    
    Args:
        image: Input image
        box: Bounding box coordinates (x, y, w, h)
        padding (float): Padding percentage to add around the bounding box
        
    Returns:
        tuple: (text, confidence)
    """
    # Extract box coordinates
    (x, y, w, h) = box
    
    # Add padding to the bounding box
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    
    # Ensure the padding doesn't go out of image bounds
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w += 2 * pad_x
    h += 2 * pad_y
    
    # Make sure the box doesn't extend beyond the image
    img_h, img_w = image.shape[:2]
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    
    # Extract the region of interest
    roi = image[y:y+h, x:x+w]
    
    # If the ROI is too small, skip it
    if roi.size == 0 or roi.shape[0] <= 10 or roi.shape[1] <= 10:
        return "", 0
    
    # Preprocess the ROI for better OCR
    roi_processed = preprocess_for_ocr(roi)
    
    # Try multiple OCR configurations
    configs = [
        "--oem 1 --psm 7",  # Treat as single line
        "--oem 1 --psm 6",  # Assume uniform block of text
        "--oem 1 --psm 8",  # Treat as single word
        "--oem 1 --psm 13"  # Treat as raw line
    ]
    
    best_text = ""
    best_conf = 0
    
    for config in configs:
        try:
            # Try with preprocessed image
            results = pytesseract.image_to_data(roi_processed, config=config, output_type=Output.DICT)
            
            # Extract the text with the highest confidence
            for i in range(len(results["text"])):
                t = results["text"][i]
                conf = int(results["conf"][i])
                
                if conf > best_conf and t.strip():
                    best_text = t
                    best_conf = conf
            
            # If we found good text, stop trying
            if best_conf > 60:
                break
                
            # If confidence is low, try with original image
            if best_conf < 40:
                # Convert to RGB for Tesseract
                if len(roi.shape) == 3:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                else:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                    
                results = pytesseract.image_to_data(roi_rgb, config=config, output_type=Output.DICT)
                
                # Check if we get better results
                for i in range(len(results["text"])):
                    t = results["text"][i]
                    conf = int(results["conf"][i])
                    
                    if conf > best_conf and t.strip():
                        best_text = t
                        best_conf = conf
                
                if best_conf > 60:
                    break
            
        except Exception as e:
            print(f"Error in OCR recognition with config {config}: {e}")
            continue
    
    return best_text, best_conf / 100.0  # Normalize confidence to [0, 1]

def detect_text(image_path, min_conf=0, model_path=None):
    """
    Detect and recognize text in an image using EAST for detection and Tesseract for recognition.
    
    Args:
        image_path (str): Path to the input image
        min_conf (float): Minimum confidence threshold (0-100) for text recognition
        model_path (str, optional): Path to the pre-trained EAST model
        
    Returns:
        list: List of detected text tokens with confidence above the threshold
    """
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Load the EAST model
    try:
        net = load_east_model(model_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        # Create a directory for models if it doesn't exist
        os.makedirs("models", exist_ok=True)
        # Provide instructions for downloading the model manually
        print("Please download the EAST model and place it in the models directory:")
        print("https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb")
        return []
    
    # Try different detection parameters
    # First try with higher confidence for better precision
    boxes = east_detect_text_regions(image, net, min_confidence=0.5)
    
    # If we don't find enough boxes, try with lower confidence
    if len(boxes) < 3:
        boxes = east_detect_text_regions(image, net, min_confidence=0.3)
    
    # If still not enough, try with even lower confidence
    if len(boxes) < 2:
        boxes = east_detect_text_regions(image, net, min_confidence=0.2)
    
    # Recognize text in each detected region
    detected_text = []
    min_conf_normalized = min_conf / 100.0  # Convert to range [0, 1]
    
    for box in boxes:
        text, confidence = recognize_text(image, box)
        
        # Add text to the list if confidence is above the threshold
        # and text is not empty
        if confidence > min_conf_normalized and text.strip():
            # Clean up the text (remove non-ASCII characters)
            cleaned_text = cleanup_text(text)
            if cleaned_text:  # Only add if text is not empty after cleanup
                detected_text.append(cleaned_text)
    
    # If we don't have much text, try using Tesseract on the whole image as a fallback
    if len(detected_text) < 2:
        try:
            # Preprocess the whole image
            preprocessed = preprocess_for_ocr(image)
            
            # Use Tesseract to detect text in the whole image
            results = pytesseract.image_to_data(preprocessed, output_type=Output.DICT)
            
            # Extract text with confidence above threshold
            for i in range(len(results["text"])):
                text = results["text"][i]
                conf = int(results["conf"][i]) / 100.0  # Normalize to [0, 1]
                
                if conf > min_conf_normalized and text.strip():
                    cleaned_text = cleanup_text(text)
                    if cleaned_text and cleaned_text not in detected_text:
                        detected_text.append(cleaned_text)
        except Exception as e:
            print(f"Error in fallback text detection: {e}")
    
    return detected_text

def process_image_file(image_path, min_conf=0, model_path=None):
    """
    Process an image file and return detected text as a dictionary
    compatible with the evaluation function.
    
    Args:
        image_path (str): Path to the input image
        min_conf (int): Minimum confidence threshold for text detection
        model_path (str, optional): Path to the pre-trained EAST model
        
    Returns:
        dict: Dictionary with image_id as key and list of detected text as value
    """
    # Extract image ID from the filename (assuming filename is the ID with extension)
    import os
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Detect text in the image
    detected_text = detect_text(image_path, min_conf, model_path)
    
    # Return dictionary with image_id as key and detected text as value
    return {image_id: detected_text}

def process_directory(directory_path, min_conf=0, output_file=None, model_path=None):
    """
    Process all images in a directory and return detected text for all images.
    
    Args:
        directory_path (str): Path to directory containing images
        min_conf (int): Minimum confidence threshold for text detection
        output_file (str, optional): Path to save results as JSON
        model_path (str, optional): Path to the pre-trained EAST model
        
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
    
    # Load EAST model once to reuse for all images
    try:
        net = load_east_model(model_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        # Create a directory for models if it doesn't exist
        os.makedirs("models", exist_ok=True)
        # Provide instructions for downloading the model manually
        print("Please download the EAST model and place it in the models directory:")
        print("https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb")
        return results
    
    for filename in os.listdir(directory_path):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            image_path = os.path.join(directory_path, filename)
            image_id = os.path.splitext(filename)[0]
            try:
                # Use the already loaded model for detection
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image from {image_path}")
                    continue
                
                # Try different detection parameters
                # First try with higher confidence for better precision
                boxes = east_detect_text_regions(image, net, min_confidence=0.5)
                
                # If we don't find enough boxes, try with lower confidence
                if len(boxes) < 3:
                    boxes = east_detect_text_regions(image, net, min_confidence=0.3)
                
                # If still not enough, try with even lower confidence
                if len(boxes) < 2:
                    boxes = east_detect_text_regions(image, net, min_confidence=0.2)
                
                # Recognize text in each detected region
                detected_text = []
                min_conf_normalized = min_conf / 100.0  # Convert to range [0, 1]
                
                for box in boxes:
                    text, confidence = recognize_text(image, box)
                    
                    # Add text to the list if confidence is above the threshold
                    # and text is not empty
                    if confidence > min_conf_normalized and text.strip():
                        # Clean up the text (remove non-ASCII characters)
                        cleaned_text = cleanup_text(text)
                        if cleaned_text:  # Only add if text is not empty after cleanup
                            detected_text.append(cleaned_text)
                
                # If we don't have much text, try using Tesseract on the whole image as a fallback
                if len(detected_text) < 2:
                    try:
                        # Preprocess the whole image
                        preprocessed = preprocess_for_ocr(image)
                        
                        # Use Tesseract to detect text in the whole image
                        results_tess = pytesseract.image_to_data(preprocessed, output_type=Output.DICT)
                        
                        # Extract text with confidence above threshold
                        for i in range(len(results_tess["text"])):
                            text = results_tess["text"][i]
                            conf = int(results_tess["conf"][i]) / 100.0  # Normalize to [0, 1]
                            
                            if conf > min_conf_normalized and text.strip():
                                cleaned_text = cleanup_text(text)
                                if cleaned_text and cleaned_text not in detected_text:
                                    detected_text.append(cleaned_text)
                    except Exception as e:
                        print(f"Error in fallback text detection: {e}")
                
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

if __name__ == "__main__":
    import argparse
    
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image to localize")
    ap.add_argument("-d", "--dir", help="path to directory of images to process")
    ap.add_argument("-c", "--min-conf", type=int, default=0,
        help="minimum confidence value to filter weak text detection")
    ap.add_argument("-o", "--output", help="path to save output JSON file")
    ap.add_argument("-m", "--model", help="path to EAST text detection model")
    args = vars(ap.parse_args())
    
    # Check if image or directory is specified
    if args["image"]:
        # Process single image
        result = process_image_file(args["image"], args["min_conf"], args["model"])
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
        process_directory(args["dir"], args["min_conf"], args["output"], args["model"])
    else:
        print("Please specify either an image file (-i) or directory (-d)")
