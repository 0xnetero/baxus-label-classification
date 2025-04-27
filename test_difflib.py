from src.wine_names import WINE_NAMES

def find_closest_wine_name(words):
    """
    Find the closest matching wine name from a list of input words.
    
    Args:
        words: List of strings to check against known wine names
        
    Returns:
        String representing the closest matching wine name
    """
    import difflib
    
    
    # Find the best match for each word against the wine names
    best_matches = []
    for word in words:
        # Get closest matches for this word
        matches = difflib.get_close_matches(word, WINE_NAMES, n=1, cutoff=0.6)
        if matches:
            best_matches.append((matches[0], difflib.SequenceMatcher(None, word, matches[0]).ratio()))
    
    # If no good matches were found
    if not best_matches:
        # Try matching combined words against wine names
        combined_text = " ".join(words)
        best_wine = None
        best_ratio = 0
        
        for wine in WINE_NAMES:
            ratio = difflib.SequenceMatcher(None, combined_text, wine).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_wine = wine
                
        return best_wine
    
    # Return the wine name with highest similarity score
    best_matches.sort(key=lambda x: x[1], reverse=True)
    return best_matches[0][0]

# Example usage
if __name__ == "__main__":
    print(find_closest_wine_name(["FourRoses", "Bourbon"]))  # Should return "Four Roses Bourbon"

    # should return "Isaac Bowman Port Barrel Finish Bourbon"
    print(find_closest_wine_name(['|', 'ISAAC', 'BOWMAN', 'SPIRIT.@', 'STRAIGHT', 'BOURBON', 'WHISKEY', '—', 'FINISHED', 'IN', 'PORT', 'BARRELS', '46%', 'ALC/VOL', '(92', 'PROOF)', '=', '!', 'S:', '\\', 'if', ':']))