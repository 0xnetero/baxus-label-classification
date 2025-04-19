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

### Run comparative testing

```bash
python test_east_accuracy.py --max-images <num_images> --threads <num_threads>
```

## Evaluation Results

When testing on a small set of images, we found:

1. **Standard Tesseract OCR**:
   - detector confidence: 80%
   - accuracy: 7.58%
   - ocr time: 98s
   - eval time: 9.6s

2. **EAST + Tesseract OCR**:
   - detector confidence: 80%
   - accuracy: 12.05%
   - ocr time: 2487s
   - eval time: 8.5s

Based on our experiments, the standard Tesseract approach currently produces better recognition results overall, but the EAST detector shows potential for detecting text in more challenging images.