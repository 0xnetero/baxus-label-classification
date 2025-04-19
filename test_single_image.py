#!/usr/bin/env python3

import os
import cv2
import sys
import json
import numpy as np
from src.east import detect_text as east_detect_text
from src.east import east_detect_text_regions, recognize_text
from src.tesseract import detect_text as tesseract_detect_text
from src.tesseract import get_text_boxes
from src.helpers import find_best_wine_match
from src.eval import evaluate_ocr_accuracy

def visualize_boxes(image, boxes, texts, title="Text Detection", box_color=(0, 255, 0), text_color=(0, 0, 255)):
    """
    Visualize bounding boxes and recognized text on the image.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [x, y, w, h]
        texts: List of text strings for each box
        title: Window title
        box_color: Color for the bounding boxes
        text_color: Color for the text
    """
    # Make a copy of the image to avoid modifying the original
    vis_image = image.copy()
    
    # Upscale image for better visualization (2x)
    h, w = vis_image.shape[:2]
    vis_image = cv2.resize(vis_image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    
    # Adjust box coordinates for the upscaled image
    upscaled_boxes = []
    for box in boxes:
        x, y, w, h = box
        upscaled_boxes.append([x*2, y*2, w*2, h*2])
    
    # Draw boxes and text
    for i, (box, text) in enumerate(zip(upscaled_boxes, texts)):
        x, y, w, h = box
        # Draw rectangle with thicker lines for better visibility
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), box_color, 3)
        
        # Prepare text (add box number and confidence if available)
        display_text = f"{i+1}: {text}"
        
        # Put text above the rectangle with larger font for better readability
        cv2.putText(vis_image, display_text, (x, max(y - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    # Scale image if too large for display
    max_height = 1000
    max_width = 1500
    
    h, w = vis_image.shape[:2]
    if h > max_height or w > max_width:
        scale = min(max_height / h, max_width / w)
        vis_image = cv2.resize(vis_image, None, fx=scale, fy=scale)
    
    # Display image
    try:
        cv2.imshow(title, vis_image)
        cv2.waitKey(0)
    except Exception as e:
        print(f"Error displaying image: {e}")
        print("Saving visualization without displaying...")
    
    return vis_image

def save_visualization(image, filename):
    """Save the visualization image to a file"""
    os.makedirs("visualizations", exist_ok=True)
    output_path = os.path.join("visualizations", filename)
    
    # Save with high quality (95% JPEG quality)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Visualization saved to {output_path}")
    
    # Also save a PNG version for best quality
    png_path = os.path.splitext(output_path)[0] + ".png"
    cv2.imwrite(png_path, image)
    print(f"PNG visualization saved to {png_path}")

def run_test_on_image(image_path):
    """
    Test both OCR methods on a single image and compare results.
    
    Args:
        image_path (str): Path to the image file
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return
    
    # Get image ID from filename
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Check if the image ID exists in dataset.csv
    import csv
    ground_truth = None
    try:
        with open('dataset.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('id') == image_id:
                    ground_truth = row.get('name', '')
                    break
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        
    if not ground_truth:
        print(f"Warning: No ground truth found for image ID {image_id} in dataset.csv.")
    else:
        print(f"Ground Truth: {ground_truth}")
    
    # Display the image
    print(f"Image: {image_path}")
    print(f"Size: {os.path.getsize(image_path) / 1024:.1f} KB")
    
    # Load image for visualization
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    
    h, w, _ = image.shape
    print(f"Dimensions: {w}x{h} pixels")
    
    # Test Tesseract OCR
    print("\n=== Tesseract OCR ===")
    conf_thresholds = [0, 20, 40, 60, 80]
    
    for conf in conf_thresholds:
        print(f"\nConfidence threshold: {conf}")
        
        # Get bounding boxes and text from Tesseract
        tesseract_boxes, tesseract_texts, tesseract_confs = get_text_boxes(image_path, conf)
        print(f"Detected {len(tesseract_boxes)} text regions")
        
        # Print details of each detected box
        for i, (box, text, conf_val) in enumerate(zip(tesseract_boxes, tesseract_texts, tesseract_confs)):
            print(f"Box {i+1}: {text} (Confidence: {conf_val:.2f})")
        
        # Get the list of tokens for matching
        tesseract_results = tesseract_detect_text(image_path, conf)
        print(f"Detected text tokens: {tesseract_results}")
        
        # Find best wine match if we have tokens
        if tesseract_results:
            best_match, confidence = find_best_wine_match(tesseract_results, min_score=0.5)
            print(f"Best match: {best_match}")
            print(f"Confidence: {confidence:.2f}")
            
            # Check if match is correct
            if ground_truth:
                normalized_match = best_match.lower().replace("'", "").replace("'", "").strip() if best_match else ""
                normalized_truth = ground_truth.lower().replace("'", "").replace("'", "").strip()
                is_correct = normalized_match == normalized_truth
                print(f"Correct: {'✓' if is_correct else '✗'}")
        else:
            print("No text detected.")
        
        # Visualize boxes if there are any
        if tesseract_boxes:
            tesseract_vis = visualize_boxes(
                image, 
                tesseract_boxes, 
                tesseract_texts, 
                f"Tesseract OCR (conf={conf})", 
                box_color=(0, 255, 0),
                text_color=(0, 0, 255)
            )
            # Save visualization
            save_visualization(tesseract_vis, f"tesseract_vis_{image_id}_conf{conf}.jpg")
    
    # Test EAST OCR
    print("\n=== EAST OCR ===")
    
    try:
        # Try to load the EAST model
        from src.east import load_east_model
        net = load_east_model()
        
        for conf in conf_thresholds:
            print(f"\nConfidence threshold: {conf}")
            
            # Get text regions using EAST
            east_boxes = east_detect_text_regions(image, net, min_confidence=conf/100)
            print(f"Detected {len(east_boxes)} text regions")
            
            # Recognize text in each region
            east_texts = []
            east_confs = []
            
            for i, box in enumerate(east_boxes):
                text, conf_val = recognize_text(image, box)
                east_texts.append(text)
                east_confs.append(conf_val)
                print(f"Box {i+1}: {text} (Confidence: {conf_val:.2f})")
            
            # Get tokens for matching
            east_results = east_detect_text(image_path, conf)
            print(f"Detected text tokens: {east_results}")
            
            # Find best wine match if we have tokens
            if east_results:
                best_match, confidence = find_best_wine_match(east_results, min_score=0.5)
                print(f"Best match: {best_match}")
                print(f"Confidence: {confidence:.2f}")
                
                # Check if match is correct
                if ground_truth:
                    normalized_match = best_match.lower().replace("'", "").replace("'", "").strip() if best_match else ""
                    normalized_truth = ground_truth.lower().replace("'", "").replace("'", "").strip()
                    is_correct = normalized_match == normalized_truth
                    print(f"Correct: {'✓' if is_correct else '✗'}")
            else:
                print("No text detected.")
            
            # Visualize boxes if there are any
            if east_boxes:
                east_vis = visualize_boxes(
                    image, 
                    east_boxes, 
                    east_texts, 
                    f"EAST OCR (conf={conf})", 
                    box_color=(255, 0, 0),
                    text_color=(255, 255, 0)
                )
                # Save visualization
                save_visualization(east_vis, f"east_vis_{image_id}_conf{conf}.jpg")
    
    except Exception as e:
        print(f"Error in EAST detection: {e}")
    
    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    run_test_on_image(image_path) 