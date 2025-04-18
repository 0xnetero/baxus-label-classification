import os
import csv
import requests
from urllib.parse import urlparse
import time
from pathlib import Path

def download_image(url, save_path):
    """Download an image from a URL and save it to the specified path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Extract the file extension from the URL or default to .jpg
        parsed_url = urlparse(url)
        file_extension = os.path.splitext(parsed_url.path)[1]
        if not file_extension:
            file_extension = '.jpg'
        
        # Create the full save path with extension
        full_save_path = save_path + file_extension
        
        # Save the image
        with open(full_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {os.path.basename(full_save_path)}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Create images directory if it doesn't exist
    image_dir = Path('images')
    image_dir.mkdir(exist_ok=True)
    
    # Read the dataset.csv file
    with open('dataset.csv', 'r') as f:
        reader = csv.DictReader(f)
        
        # Track download statistics
        total_rows = 0
        successful_downloads = 0
        failed_downloads = 0
        
        for row in reader:
            total_rows += 1
            image_url = row.get('image_url', '')
            
            if not image_url:
                print(f"No image URL for row {total_rows}")
                failed_downloads += 1
                continue
            
            # Create filename using the ID for uniqueness
            image_id = row.get('id', str(total_rows))
            save_path = image_dir / f"{image_id}"
            
            # Download the image
            if download_image(image_url, str(save_path)):
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(0.1)
    
    # Print summary
    print("\nDownload Summary:")
    print(f"Total rows processed: {total_rows}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")

if __name__ == "__main__":
    main() 