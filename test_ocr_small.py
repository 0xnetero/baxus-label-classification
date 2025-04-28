#!/usr/bin/env python3
"""
Test script to run OCR evaluation on a small number of images.
This helps verify the timeout mechanism works correctly.
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Create test output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Run evaluation with only 10 images to test
    print("Running Tesseract OCR test on 10 images...")
    cmd = [
        "python", "test_east_accuracy.py",
        "--tesseract-only",
        "--max-images", "10",
        "--conf-thresholds", "0"  # Only test with 0 confidence threshold
    ]
    
    subprocess.run(cmd)
    print("\nTest complete! Check for any timeouts or hangs.") 