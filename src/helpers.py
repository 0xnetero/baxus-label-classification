from src.constants import DIFFLIB_CUTOFF

def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

from src.wine_names import WINE_NAMES, LOWER_CASE_WINE_NAMES
import re
from difflib import SequenceMatcher, get_close_matches
# Helper functions for the improved wine matching
def calculate_token_match_score(tokens, target_text):
    """
    Calculate a score based on how well individual tokens match the target text.
    
    Args:
        tokens (list): List of token strings
        target_text (str): The target text to match against
        
    Returns:
        float: Score between 0 and 1 representing token match quality
    """
    target_words = target_text.split()
    
    # Track matches for each token
    token_scores = []
    for token in tokens:
        # Check if token is a complete word in target (exact match)
        if token in target_words:
            token_scores.append(1.0)
        # Check if token is a substring of target
        elif token in target_text:
            token_scores.append(0.8)
        else:
            # Get best fuzzy match for this token
            best_word_match = max([SequenceMatcher(None, token, word).ratio() for word in target_words], default=0)
            token_scores.append(best_word_match)
    
    # Average all token scores
    return sum(token_scores) / len(token_scores) if token_scores else 0

def calculate_match_bonuses(tokens, target_text, original_wine):
    """
    Calculate bonus scores for specific patterns in the match.
    
    Args:
        tokens (list): List of token strings
        target_text (str): The normalized target text
        original_wine (str): The original wine name with formatting
        
    Returns:
        float: Bonus score between 0 and 1
    """
    bonus = 0
    
    # Check for numerical matches (years, etc.)
    token_numbers = [int(re.search(r'\d+', t).group()) for t in tokens if re.search(r'\d+', t)]
    target_numbers = [int(n) for n in re.findall(r'\d+', target_text)]
    
    # Bonus for matching numbers
    if token_numbers and target_numbers and any(n in target_numbers for n in token_numbers):
        bonus += 0.25
    
    # Bonus for "year" keyword
    if any("year" in t for t in tokens) and "year" in target_text:
        bonus += 0.15
    
    # Bonus for "small batch" phrase
    if any("small" in t for t in tokens) and any("batch" in t for t in tokens) and "small batch" in target_text:
        bonus += 0.2
    
    # Bonus for brand keywords
    important_brands = ["buffalo", "weller", "blanton", "stagg", "bourbon", "whiskey", "whisky"]
    for brand in important_brands:
        if any(brand in t for t in tokens) and brand in target_text:
            bonus += 0.15
            break
    
    # Specific bonus for "Original"
    if any("original" in t for t in tokens) and "original" in target_text:
        bonus += 0.15
    
    # Cap the bonus at 0.5
    return min(0.5, bonus)

def remove_special_characters(input_string):
    """Removes special characters from a string.

    Args:
        input_string: The string to process.

    Returns:
        A new string with special characters removed.
    """
    # Define the pattern of characters to remove.
    # We are using a regular expression here.
    # [^a-zA-Z0-9\s] means: match any character that is NOT
    # (^) a lowercase letter (a-z), an uppercase letter (A-Z),
    # a digit (0-9), or a whitespace character (\s).
    pattern = r'[^a-zA-Z0-9\s]'

    # Use the re.sub() function to replace all matches of the pattern
    # with an empty string ('').
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string

def find_best_wine_match(tokens, min_score=DIFFLIB_CUTOFF):
    """
    Find the best matching wine name from a list of text tokens using difflib.
    
    Args:
        tokens (list): List of strings/tokens to match against wine names
        min_score (float): Minimum similarity score to consider a match (default: 0.4)
        
    Returns:
        tuple: (best_match, confidence_score) or (None, 0) if no good match found
    """
    if not tokens or len(tokens) == 0:
        return (None, 0)
    
    # Try matching combined words against wine names
    combined_text = " ".join(tokens)
    combined_text = remove_special_characters(combined_text)
    combined_text = combined_text.lower()
    best_wine = None
    best_ratio = 0
    # print(f"Combined text: {combined_text}")
    
    for wine in LOWER_CASE_WINE_NAMES:
        ratio = SequenceMatcher(None, combined_text, wine).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_wine = wine
    
    # Return best match with confidence score or (None, 0) if no match found
    if best_wine and best_ratio >= min_score:
        return WINE_NAMES[LOWER_CASE_WINE_NAMES.index(best_wine)], best_ratio
    else:
        return None, 0