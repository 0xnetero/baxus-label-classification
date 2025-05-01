## Wine Label Recognition

This project uses OCR (Optical Character Recognition) to detect and recognize text from wine label images, and matches the detected text to known wine names.

## Installation on Ubuntu

### Install tesseract

```bash
sudo apt install tesseract-ocr
```

### Install python packages

```bash
pip install -r requirements.txt
```

### Instal paddleocr (i'm using CPU)

```bash
python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ 
```

## Implementation

The project includes two text detection methods:

1. **Standard Tesseract OCR** (`src/tesseract.py`): Uses Tesseract OCR directly on the image to detect and recognize text.
2. **EAST + Tesseract OCR** (`src/east.py`): Uses OpenCV's EAST deep learning text detector to first locate text regions, then applies Tesseract OCR to recognize text within those regions.

## Usage

### Process a single image

```bash
python -m src.tesseract -i <image_path> -c <min_confidence>
python -m src.east -i <image_path> -c <min_confidence>
```

### Process a directory of images

```bash
python -m src.tesseract -d <directory_path> -c <min_confidence> -o <output_file>
python -m src.east -d <directory_path> -c <min_confidence> -o <output_file>
```

### Evaluate OCR results

```bash
python -m src.eval <ocr_results_file> [output_file] [min_confidence] [verbose]
```

### Test a single image with both methods

```bash
python test_single_image.py <image_path>
```

### Run evaluation

```bash
python test_east_accuracy.py --tesseract-only
python test_east_accuracy.py --east-only
python test_east_accuracy.py --threads 4
```

## Evaluation Results

When testing on a small set of images, we found:

1. **Standard Tesseract OCR**:
=== SUMMARY: Standard Tesseract OCR ===
Confidence threshold | Accuracy | Correct/Total matches
------------------------------------------------------------
                   0 |     2.20% | 11/500
                  20 |     4.20% | 21/500
                  40 |     5.20% | 26/500
                  60 |     5.00% | 25/500
                  80 |     3.80% | 19/500

2. **EAST + Tesseract OCR**:
   - detector confidence: 80%
   - accuracy: 12.05%
   - ocr time: 2487s
   - eval time: 8.5s

Based on our experiments, the standard Tesseract approach currently produces better recognition results overall, but the EAST detector shows potential for detecting text in more challenging images.