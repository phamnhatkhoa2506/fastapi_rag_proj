import os
import wget
import json
import ssl
import urllib3
import re

from typing import Dict

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context


def sanitize_filename(filename: str) -> str:
    """
        Sanitize filename by removing or replacing invalid characters
    """

    # Replace invalid characters with underscore
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)

    # Remove any leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')

    return sanitized


def is_exist(file_link: str, dir_name: str) -> bool:
    '''
        Check if the file already exists
    '''

    sanitized_title = sanitize_filename(file_link["title"])
    return os.path.exists(f"./{dir_name}/{sanitized_title}.pdf")


def load_links_file(links_file_path: str) -> Dict[str, str]:
    with open(links_file_path, "r", encoding="utf-8") as f:
        file_links = json.load(f)

    return file_links


def download(dir_name: str, links: Dict[str, str]) -> None:
    for link in links:
        if not is_exist(link):
            sanitized_title = sanitize_filename(link["title"])
            wget.download(link["url"], out=f"./{dir_name}/{sanitized_title}.pdf")
    

if __name__ == '__main__':
    links_file = load_links_file('./links.json')
    print(links_file)

    download(links_file)