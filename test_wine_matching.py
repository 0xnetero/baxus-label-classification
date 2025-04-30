#!/usr/bin/env python3

from src.helpers import find_best_wine_match

def test_specific_examples():
    print("Testing specific problematic examples:")
    print("-" * 50)
    
    # Test case for Macallan 12 Year Double Cask
    tokens = ['MACALLAN', 'HIGHLAND SINGLE MALT', '12', '12', 'DOUBLE CASK', 
              'MACALLAN', 'HIGHLAND SINGLE', 'SCOTCH WHISKY', '12', 'DOUBLE CASK', 'HAND', 'LTD']
    expected = "Macallan 12 Year Double Cask"
    
    match, confidence = find_best_wine_match(tokens)
    
    print(f"Input tokens: {tokens}")
    print(f"Expected match: {expected}")
    print(f"Actual match: {match}")
    print(f"Confidence score: {confidence:.4f}")
    print(f"Success: {'✓' if match == expected else '✗'}")
    
    # Test case for typical lower confidence but correct match
    tokens2 = ['Buffalo', 'Trace', 'Kentucky', 'Straight', 'Bourbon', 'Whiskey']
    expected2 = "Buffalo Trace"
    
    match2, confidence2 = find_best_wine_match(tokens2)
    
    print("\n" + "-" * 50)
    print(f"Input tokens: {tokens2}")
    print(f"Expected match: {expected2}")
    print(f"Actual match: {match2}")
    print(f"Confidence score: {confidence2:.4f}")
    print(f"Success: {'✓' if match2 == expected2 else '✗'}")
    
    # Test case for missing the "Year" word
    tokens3 = ['EAGLE', 'RARE', '10', 'BOURBON', 'WHISKEY']
    expected3 = "Eagle Rare 10 Year"
    
    match3, confidence3 = find_best_wine_match(tokens3)
    
    print("\n" + "-" * 50)
    print(f"Input tokens: {tokens3}")
    print(f"Expected match: {expected3}")
    print(f"Actual match: {match3}")
    print(f"Confidence score: {confidence3:.4f}")
    print(f"Success: {'✓' if match3 == expected3 else '✗'}")

if __name__ == "__main__":
    test_specific_examples() 