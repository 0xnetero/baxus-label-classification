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
    
    # Clean up and deduplicate tokens
    cleaned_tokens = []
    for token in tokens:
        # Convert to lowercase and remove special characters
        clean_token = remove_special_characters(token.lower())
        if clean_token and clean_token not in cleaned_tokens:
            cleaned_tokens.append(clean_token)
    
    # Try direct matching with subsets of tokens
    best_wine = None
    best_ratio = 0
    
    # Extract numbers from tokens
    numbers = [token for token in cleaned_tokens if token.isdigit()]
    
    # Method 1: Standard combined text approach
    combined_text = " ".join(cleaned_tokens)
    
    for wine in LOWER_CASE_WINE_NAMES:
        ratio = SequenceMatcher(None, combined_text, wine).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_wine = wine
    
    # Save the best difflib match
    difflib_best_wine = best_wine
    difflib_best_ratio = best_ratio
    
    # Method 2: Check for key tokens match (brand names and numbers)
    for wine_idx, wine in enumerate(LOWER_CASE_WINE_NAMES):
        wine_tokens = wine.split()
        
        # Extract numbers from wine name
        wine_numbers = [word for word in wine_tokens if word.isdigit()]
        
        # Calculate how many words from the wine name are found in our tokens
        matches = 0
        wine_words_count = len(wine_tokens)
        
        # Special handling for year/number tokens
        year_in_wine = False
        year_in_tokens = False
        matching_number = False
        
        for wine_word in wine_tokens:
            # Check if this word is a number (likely a year or age)
            if wine_word.isdigit():
                year_in_wine = True
                year_value = wine_word
                
                # Look for matching numbers in tokens
                for token in cleaned_tokens:
                    if token.isdigit() and token == year_value:
                        matching_number = True
                        year_in_tokens = True
                        matches += 1
                        break
            else:
                # For regular words, check if they appear in our tokens
                for token in cleaned_tokens:
                    if wine_word in token or token in wine_word:
                        matches += 1
                        break
        
        # Calculate match ratio for this wine
        # Give more weight to wines where we match most words
        token_match_ratio = matches / wine_words_count if wine_words_count > 0 else 0
        
        # Add bonus for matching the brand/first word
        brand_bonus = 0.1 if wine_tokens and wine_tokens[0] in " ".join(cleaned_tokens).lower() else 0
        
        # Add bonus for matching year numbers (important for aged products)
        year_bonus = 0.15 if matching_number else 0
        
        # Add word-matching bonus for "year" if missing but implied by number matching
        year_word_bonus = 0
        if "year" in wine.lower() and not any("year" in t.lower() for t in cleaned_tokens):
            if year_in_wine and year_in_tokens:
                # If wine has "X Year" and tokens have the number X but not "year"
                year_word_bonus = 0.2
        
        # Special case: Match wines with identical numbers
        number_match_bonus = 0
        if numbers and wine_numbers and set(numbers).intersection(set(wine_numbers)):
            number_match_bonus = 0.2
            
            # Strong preference for age-statement match with the exact number
            if len(wine.split()) >= 3 and "year" in wine.lower():
                # Products with age statements (like "10 Year") should match better when numbers are present
                number_match_bonus = 0.4
        
        # Calculate final score for this wine
        wine_score = token_match_ratio + brand_bonus + year_bonus + year_word_bonus + number_match_bonus
        
        # If this score is better, update best match
        if wine_score > best_ratio:
            best_ratio = wine_score
            best_wine = LOWER_CASE_WINE_NAMES[wine_idx]
    
    # If token matching didn't find a strong match but difflib did, use the difflib result
    if best_ratio < 0.5 and difflib_best_ratio >= min_score:
        best_wine = difflib_best_wine
        best_ratio = difflib_best_ratio
    
    # Return best match with confidence score or (None, 0) if no match found
    if best_wine and best_ratio >= min_score:
        return WINE_NAMES[LOWER_CASE_WINE_NAMES.index(best_wine)], best_ratio
    else:
        return None, 0

# Import fuzzywuzzy for improved fuzzy matching
try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("Warning: fuzzywuzzy package not found. Fuzzy matching will not be available.")
    print("Install with: pip install fuzzywuzzy python-Levenshtein")

def find_best_wine_match_fuzzy(tokens, min_score=60):
    """
    Find the best matching wine name from a list of text tokens using fuzzywuzzy.
    
    Args:
        tokens (list): List of strings/tokens to match against wine names
        min_score (int): Minimum similarity score to consider a match (default: 60)
        
    Returns:
        tuple: (best_match, confidence_score) or (None, 0) if no good match found
    """
    if not tokens or len(tokens) == 0:
        return (None, 0)
    
    if not FUZZYWUZZY_AVAILABLE:
        # Fall back to the standard matching function if fuzzywuzzy is not available
        return find_best_wine_match(tokens, DIFFLIB_CUTOFF)
    
    # Clean up and deduplicate tokens
    cleaned_tokens = []
    for token in tokens:
        # Convert to lowercase and remove special characters
        clean_token = remove_special_characters(token.lower())
        if clean_token and clean_token not in cleaned_tokens:
            cleaned_tokens.append(clean_token)
    
    # Extract numbers from tokens (for age statement matching)
    numbers = [token for token in cleaned_tokens if token.isdigit()]
    
    # Join tokens into a single string for full text matching
    combined_text = " ".join(cleaned_tokens)
    
    # Method 1: Try direct matching with fuzzywuzzy process.extractOne
    match_result = process.extractOne(
        combined_text, 
        WINE_NAMES,
        scorer=fuzz.token_sort_ratio
    )
    
    if match_result:
        best_match, score = match_result
    else:
        best_match, score = None, 0
    
    # Method 2: Try partial ratio matching (better for matching substrings)
    partial_match_result = process.extractOne(
        combined_text, 
        WINE_NAMES,
        scorer=fuzz.partial_ratio
    )
    
    if partial_match_result and partial_match_result[1] > score:
        best_match, score = partial_match_result
    
    # Method 3: Use token set ratio (better for handling extra words and word order)
    token_set_match_result = process.extractOne(
        combined_text, 
        WINE_NAMES,
        scorer=fuzz.token_set_ratio
    )
    
    if token_set_match_result and token_set_match_result[1] > score:
        best_match, score = token_set_match_result
    
    # Store the initial best match
    initial_best_match = best_match

    # Special handling for age statements - look for wines with the same number (age) as in tokens
    if numbers:
        # Find wines containing the same numbers as in our tokens
        matching_number_wines = []
        for wine in WINE_NAMES:
            wine_numbers = [n for n in re.findall(r'\d+', wine)]
            if any(num in numbers for num in wine_numbers):
                matching_number_wines.append((wine, fuzz.token_set_ratio(combined_text, wine)))
        
        # Sort by similarity score
        matching_number_wines.sort(key=lambda x: x[1], reverse=True)
        
        # If we have any matches with the same number, use the best one
        if matching_number_wines and matching_number_wines[0][1] >= min_score:
            age_match, age_score = matching_number_wines[0]
            
            # Prefer matches with "Year" if a number is present
            if "year" in age_match.lower() and any(num in age_match for num in numbers):
                best_match = age_match
                score = age_score + 10  # Add bonus for year match
    
    # Add bonus score for matching numbers (age statements)
    bonus_score = 0
    if best_match and numbers:
        # Extract numbers from the best match
        match_numbers = [n for n in re.findall(r'\d+', best_match)]
        
        # Add bonus for matching numbers (important for age statements)
        if match_numbers and any(num in numbers for num in match_numbers):
            bonus_score += 10
            
            # Extra bonus if the match contains "Year" (age statement)
            if "year" in best_match.lower():
                bonus_score += 10
    
    # Add bonus for exact brand name match
    if best_match:
        brand_name = best_match.split()[0].lower()
        if any(brand_name == token.lower() for token in cleaned_tokens):
            bonus_score += 10
    
    # Special case for "Eagle Rare 10 Year" vs "Eagle Rare"
    # If we have a number in tokens and we're matching a short name (Eagle Rare),
    # check if there's a variant with the number and "Year" in it
    if best_match and numbers and "year" not in best_match.lower():
        # Look for a variant with same brand but including our number and "Year"
        brand_prefix = " ".join(best_match.split()[:2]).lower()  # First two words as brand
        
        for wine in WINE_NAMES:
            wine_lower = wine.lower()
            if wine_lower.startswith(brand_prefix) and "year" in wine_lower:
                # Check if our detected number is in this wine name
                if any(num in re.findall(r'\d+', wine) for num in numbers):
                    best_match = wine
                    bonus_score += 20
                    break
    
    # Apply bonus score and ensure it doesn't exceed 100
    final_score = min(score + bonus_score, 100)
    
    # Return the best match if score is high enough
    if best_match and final_score >= min_score:
        return best_match, final_score / 100.0
    else:
        return None, 0