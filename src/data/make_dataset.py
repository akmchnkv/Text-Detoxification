import os
import requests
import pandas as pd
from zipfile import ZipFile

# Constants
DATA_URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
DATA_DIR = "../data/raw"
ZIP_FILE_PATH = os.path.join(DATA_DIR, "filtered_paranmt.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "filtered_paranmt")

def download_data(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with requests.get(url, stream=True) as response:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

def extract_data(zip_file, extract_path):
    with ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data

def save_processed_data(data, save_path):
    data.to_csv(save_path, index=False)

def main():
    # Download and extract data
    download_data(DATA_URL, ZIP_FILE_PATH)
    extract_data(ZIP_FILE_PATH, EXTRACTED_DIR)

    # Load the data
    data_path = os.path.join(EXTRACTED_DIR, "filtered_paranmt.tsv")
    dataset = load_data(data_path)

    # Process and save the data
    processed_data_path = os.path.join("../data/interim", "processed_data.csv")
    save_processed_data(dataset, processed_data_path)

if __name__ == "__main__":
    main()
