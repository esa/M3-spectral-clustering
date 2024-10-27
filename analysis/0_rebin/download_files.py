import os
import pickle
import shutil
import requests
from bs4 import BeautifulSoup
import random

target = '0004' # 0004 or 0003

# Base URLs to crawl
base_urls = [
    f'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_{target}/',
]

# Define media paths and base directory
media_paths = [
    "/media/freya/data_1/data/m3/",
    "/media/freya/data_2/data/m3/",
    "/media/freya/data_3/data/m3/"
]

urls_file_path = f'file_urls_{target}.pkl'


# Function to get available disk space in a directory
def get_available_space(directory):
    total, used, free = shutil.disk_usage(directory)
    return free

# Check if a file already exists in any of the media directories
def file_exists(file_path):
    for media_path in media_paths:
        _media_path = os.path.join(media_path, f'CH1M3_{target}')
        full_path = os.path.join(_media_path, file_path)
        if os.path.exists(full_path):
            print(f"File already exists: {full_path}")
            return True
    return False

# Find a media directory with sufficient space
def get_media_dir(min_space_required):
    for media_path in media_paths:
        if get_available_space(media_path) > min_space_required:
            print(f"Selected media directory with sufficient space: {media_path}")
            return media_path
    print("No media directory with sufficient space found.")
    return None

# Function to remove M3T* files from the directories
def remove_m3t_files():
    print("Removing M3T* files...")
    for media_path in media_paths:
        for root, dirs, files in os.walk(media_path):
            for file in files:
                if file.startswith('M3T'):
                    file_path = os.path.join(root, file)
                    print(f"Removing file: {file_path}")
                    os.remove(file_path)

# Crawl the website to get file URLs
def crawl_and_get_file_urls(base_url, visited_urls):
    file_urls = []
    print(f"Crawling {base_url}...")

    # Check if this URL was already visited
    if base_url in visited_urls:
        return []

    visited_urls.add(base_url)  # Mark this URL as visited

    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to access {base_url}. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            href = href.strip()
            print(f"Checking link: {href}")

            # Check if it's a directory
            if href.endswith('/'):
                subdirectory_url = base_url + href
                print(f"Found subdirectory: {subdirectory_url}")
                # Crawl this subdirectory
                file_urls += crawl_and_get_file_urls(subdirectory_url, visited_urls)

            # Check for the specific DATA directory structure and file types
            elif ('V03_LOC' in href or 'RFL' in href) and (href.endswith('.HDR') or href.endswith('.IMG')):
                if 'M3T' in href:
                    continue
                else:
                    file_urls.append(base_url + href)
                    print(f"Found data file: {href}")

    return file_urls

# Download a file with error handling and removal of incomplete files
import os
import requests

# Download a file with error handling, resuming partial downloads
def download_file(file_url, target_dir):
    file_name = os.path.basename(file_url)
    file_path = os.path.join(target_dir, file_name)

    try:
        # Check if the file exists and get its size
        existing_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        headers = {'Range': f'bytes={existing_size}-'}
    except OSError as e:
        print(f"Error getting file size: {e}")
        existing_size = 0
        headers = {}

    try:
        response = requests.get(file_url, stream=True, headers=headers)
        response.raise_for_status()

        with open(file_path, 'ab') as f:  # Open in append mode
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded {file_name}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_url}: {e}")
        # Do not remove the incomplete file


# Download files
def download_files():
    visited_urls = set()  # Track visited directories

    for base_url in base_urls:
        # Crawl to get the list of file URLs
        if os.path.exists(urls_file_path):
            # Load the list from the file
            with open(urls_file_path, 'rb') as f:
                file_urls = pickle.load(f)
            print("File loaded successfully.")
            #print(file_urls)
        else:
            file_urls = crawl_and_get_file_urls(base_url, visited_urls)

            if not file_urls:
                print("No files found for download.")
                continue

            # Save the list to a file using pickle
            with open(urls_file_path, 'wb') as f:
                pickle.dump(file_urls, f)
        random.shuffle(file_urls)

        for file_url in file_urls:
            relative_path = file_url.replace(base_url, "").lstrip('/')

            # Skip if file already exists
            if file_exists(relative_path):
                continue

            # Check available disk space and get the appropriate media directory
            media_dir = get_media_dir(1024 * 1024 * 100)  # Minimum space required: 100MB
            if media_dir:
                # Perform the download
                target_dir = os.path.join(os.path.join(media_dir, f'CH1M3_{target}'), os.path.dirname(relative_path))
                os.makedirs(target_dir, exist_ok=True)
                print('downloading', file_url)
                download_file(file_url, target_dir)
            else:
                print("Error: No available space in any media directories.")

if __name__ == "__main__":
    # First, remove any M3T* files from the drives
    # remove_m3t_files()

    # Then start the download process
    download_files()