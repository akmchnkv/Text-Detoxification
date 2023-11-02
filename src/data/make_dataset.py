import os
import zipfile
import pandas as pd
import requests

DATA_URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
DATA_DIR = "../data/raw"
ZIP_FILE_PATH = os.path.join(DATA_DIR, "filtered_paranmt.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "filtered_paranmt")

def download_and_extract_dataset(url, zip_file_path, extracted_dir):
    # Download the zip file
    response = requests.get(url)
    with open(zip_file_path, 'wb') as zip_file:
        zip_file.write(response.content)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

def load_dataframe(extracted_dir, tsv_file_name='filtered.tsv'):
    tsv_file_path = os.path.join(extracted_dir, tsv_file_name)

    # Check file existence and extract the dataset if necessary
    if not os.path.exists(tsv_file_path):
        download_and_extract_dataset(DATA_URL, ZIP_FILE_PATH, EXTRACTED_DIR)

    # Read the TSV file into a DataFrame and drop unnecessary columns
    dataframe = pd.read_csv(tsv_file_path, delimiter='\t')
    return dataframe)