#!/usr/bin/env python3
import json
import os
import re
from collections import Counter, defaultdict
import difflib
from fuzzywuzzy import fuzz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tabulate import tabulate

# File path for the evaluation results
EVAL_RESULTS_FILE = '/home/netero/Documents/ai/label-recognition/best_checkpoints/new_paddle_fuzzy_eval_results_conf0.8.json'
OUTPUT_DIR = 'failure_analysis'

def load_evaluation_results(file_path):
    """Load the evaluation results JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_failed_cases(results):
    """Extract all cases where is_correct is False."""
    failed_cases = []
    for result in results.get('results', []):
        if not result.get('is_correct', False):
            failed_cases.append(result)
    return failed_cases

def preprocess_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_token_presence(failed_cases):
    """Analyze which tokens from ground truth are present/missing in OCR results."""
    missing_tokens = []
    present_tokens = []
    
    for case in failed_cases:
        ground_truth = case.get('ground_truth', '')
        if not ground_truth:
            continue
            
        # Normalize and tokenize ground truth
        gt_tokens = preprocess_text(ground_truth).split()
        
        # Normalize and join OCR tokens
        ocr_text = ' '.join(case.get('ocr_tokens', []))
        ocr_text = preprocess_text(ocr_text)
        
        # Check which tokens are present/missing
        for token in gt_tokens:
            if token in ocr_text:
                present_tokens.append(token)
            else:
                missing_tokens.append(token)
    
    return Counter(missing_tokens), Counter(present_tokens)

def analyze_confidence_scores(failed_cases):
    """Analyze confidence scores of failed cases."""
    confidences = [case.get('confidence', 0) for case in failed_cases]
    return {
        'mean': np.mean(confidences),
        'median': np.median(confidences),
        'min': min(confidences),
        'max': max(confidences),
        'histogram': np.histogram(confidences, bins=10, range=(0, 1))
    }

def analyze_error_patterns(failed_cases):
    """Group failures by error pattern types."""
    error_patterns = defaultdict(list)
    
    for case in failed_cases:
        ground_truth = preprocess_text(case.get('ground_truth', ''))
        predicted = preprocess_text(case.get('predicted_match', ''))
        ocr_text = preprocess_text(' '.join(case.get('ocr_tokens', [])))
        
        if not ground_truth:
            error_patterns['no_ground_truth'].append(case)
            continue
            
        if not predicted:
            error_patterns['no_prediction'].append(case)
            continue
            
        # Check if it's a brand confusion error
        gt_brand = ground_truth.split()[0]
        pred_brand = predicted.split()[0] if predicted else ""
        
        if gt_brand != pred_brand:
            error_patterns['brand_confusion'].append(case)
            continue
            
        # Check if key words are missing in OCR tokens
        gt_tokens = set(ground_truth.split())
        ocr_tokens = set(ocr_text.split())
        
        missing_ratio = len(gt_tokens - ocr_tokens) / len(gt_tokens) if gt_tokens else 0
        
        if missing_ratio > 0.5:
            error_patterns['missing_key_words'].append(case)
            continue
            
        # Check if it's a minor variation
        similarity = fuzz.ratio(ground_truth, predicted)
        if similarity > 70:
            error_patterns['minor_variation'].append(case)
            continue
            
        # Fallback for unclassified errors
        error_patterns['other'].append(case)
    
    return error_patterns

def create_visualizations(missing_tokens, present_tokens, confidence_stats, error_patterns):
    """Create visualizations for the analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Most commonly missing tokens
    plt.figure(figsize=(12, 6))
    top_missing = missing_tokens.most_common(20)
    tokens, counts = zip(*top_missing) if top_missing else ([], [])
    plt.bar(tokens, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Most Commonly Missing Tokens')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'missing_tokens.png'))
    plt.close()
    
    # Confidence score histogram
    plt.figure(figsize=(10, 6))
    hist_data, bin_edges = confidence_stats['histogram']
    plt.bar(bin_edges[:-1], hist_data, width=0.05, alpha=0.7, align='edge')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Failed Cases')
    plt.title('Confidence Score Distribution for Failed Cases')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_histogram.png'))
    plt.close()
    
    # Error pattern distribution
    plt.figure(figsize=(10, 6))
    pattern_names = list(error_patterns.keys())
    pattern_counts = [len(cases) for cases in error_patterns.values()]
    plt.bar(pattern_names, pattern_counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Error Patterns')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_patterns.png'))
    plt.close()

def write_detailed_report(failed_cases, missing_tokens, present_tokens, confidence_stats, error_patterns):
    """Write a detailed report of the failure analysis."""
    report_path = os.path.join(OUTPUT_DIR, 'failure_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write('# OCR Failure Analysis Report\n\n')
        
        # Summary statistics
        f.write('## Summary Statistics\n\n')
        f.write(f'Total failed cases: {len(failed_cases)}\n')
        f.write(f'Average confidence score: {confidence_stats["mean"]:.2f}\n')
        f.write(f'Median confidence score: {confidence_stats["median"]:.2f}\n\n')
        
        # Error pattern distribution
        f.write('## Error Pattern Distribution\n\n')
        pattern_table = []
        for pattern, cases in error_patterns.items():
            pattern_table.append([pattern, len(cases), f"{len(cases)/len(failed_cases)*100:.1f}%"])
        f.write(tabulate(pattern_table, headers=['Pattern', 'Count', 'Percentage'], tablefmt='pipe'))
        f.write('\n\n')
        
        # Top missing tokens
        f.write('## Most Commonly Missing Tokens\n\n')
        missing_table = []
        for token, count in missing_tokens.most_common(20):
            missing_table.append([token, count])
        f.write(tabulate(missing_table, headers=['Token', 'Missing Count'], tablefmt='pipe'))
        f.write('\n\n')
        
        # Detailed examples for each error pattern
        f.write('## Example Cases by Error Pattern\n\n')
        for pattern, cases in error_patterns.items():
            f.write(f'### {pattern.replace("_", " ").title()}\n\n')
            f.write(f'Total cases: {len(cases)}\n\n')
            
            for i, case in enumerate(cases[:5]):  # Show up to 5 examples
                image_id = case.get('image_id', 'unknown')
                ground_truth = case.get('ground_truth', '')
                predicted = case.get('predicted_match', '')
                ocr_tokens = case.get('ocr_tokens', [])
                confidence = case.get('confidence', 0)
                
                f.write(f'**Example {i+1}:** {image_id}\n')
                f.write(f'- Ground Truth: {ground_truth}\n')
                f.write(f'- Predicted: {predicted}\n')
                f.write(f'- OCR Tokens: {", ".join(ocr_tokens)}\n')
                f.write(f'- Confidence: {confidence:.2f}\n\n')
            
            f.write('\n')
        
        # Recommendations
        f.write('## Recommendations for Improvement\n\n')
        
        if error_patterns.get('brand_confusion'):
            f.write('### Brand Confusion\n')
            f.write('- Increase weight for brand name matching in the fuzzy matching algorithm\n')
            f.write('- Add specific brand name detection pre-processing\n\n')
            
        if error_patterns.get('missing_key_words'):
            f.write('### Missing Key Words\n')
            f.write('- Improve OCR model to better detect small or stylized text\n')
            f.write('- Consider using image preprocessing techniques to enhance text visibility\n\n')
            
        if error_patterns.get('minor_variation'):
            f.write('### Minor Variations\n')
            f.write('- Fine-tune the string normalization process\n')
            f.write('- Add more synonym mappings for common terms (e.g., "whisky" vs "whiskey")\n\n')

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and process data
    print(f"Loading evaluation results from {EVAL_RESULTS_FILE}")
    data = load_evaluation_results(EVAL_RESULTS_FILE)
    
    print(f"Total cases in evaluation: {len(data.get('results', []))}")
    failed_cases = extract_failed_cases(data)
    print(f"Failed cases: {len(failed_cases)}")
    
    # Analyze the failed cases
    print("Analyzing token presence...")
    missing_tokens, present_tokens = analyze_token_presence(failed_cases)
    
    print("Analyzing confidence scores...")
    confidence_stats = analyze_confidence_scores(failed_cases)
    
    print("Identifying error patterns...")
    error_patterns = analyze_error_patterns(failed_cases)
    
    # Generate output
    print("Creating visualizations...")
    create_visualizations(missing_tokens, present_tokens, confidence_stats, error_patterns)
    
    print("Writing detailed report...")
    write_detailed_report(failed_cases, missing_tokens, present_tokens, confidence_stats, error_patterns)
    
    # Write the most challenging examples to a separate file
    with open(os.path.join(OUTPUT_DIR, 'challenging_cases.json'), 'w') as f:
        most_challenging = sorted(failed_cases, key=lambda x: x.get('confidence', 0), reverse=True)[:20]
        json.dump(most_challenging, f, indent=2)
    
    print(f"Analysis complete! Results saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main() 