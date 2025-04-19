import csv
from pathlib import Path

def extract_unique_names():
    """Extract unique names from the 'name' column in dataset.csv"""
    # Initialize an empty set to store unique names
    unique_names = set()
    
    # Read the dataset.csv file
    with open('dataset.csv', 'r') as f:
        reader = csv.DictReader(f)
        
        # Extract names from each row
        for row in reader:
            name = row.get('name', '').strip()
            if name:  # Only add non-empty names
                unique_names.add(name)
    
    # Convert set to sorted list for consistent output
    unique_names_list = sorted(list(unique_names))
    return unique_names_list

def save_to_config(names_list):
    """Save the names list to config.py file"""
    with open('config.py', 'w') as f:
        f.write("# List of unique wine names extracted from dataset.csv\n\n")
        f.write("WINE_NAMES = [\n")
        
        # Write each name as a string in the list
        for name in names_list:
            # Escape any quotes in the name
            escaped_name = name.replace("'", "\\'")
            f.write(f"    '{escaped_name}',\n")
        
        f.write("]\n")

def main():
    # Extract unique names
    unique_names = extract_unique_names()
    
    # Print statistics
    print(f"Extracted {len(unique_names)} unique wine names from dataset.csv")
    
    # Save to config.py
    save_to_config(unique_names)
    print(f"Saved unique names to config.py")

if __name__ == "__main__":
    main() 