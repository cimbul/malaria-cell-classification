import requests
import json
from pathlib import Path
from zipfile import ZipFile

def get_kaggle_credentials():
    cred_path = Path.home() / '.kaggle' / 'kaggle.json'
    with cred_path.open() as cred_file:
        return json.load(cred_file)

def fetch_kaggle_dataset(dataset_name, filename, dataset_version = 1, output_dir = Path('data'), credentials = None):
    if credentials is None:
        credentials = get_kaggle_credentials()
    response = requests.get(
        f'https://www.kaggle.com/api/v1/datasets/download/{dataset_name}/{filename}',
        params = {'datasetVersionNumber': dataset_version},
        auth = (credentials['username'], credentials['key']),
        stream = True,
    )
    response.raise_for_status()

    output_dir.mkdir(parents = True)
    output_path = output_dir / filename
    with output_path.open('wb') as output_file:
        for data in response.iter_content(4096):
            output_file.write(data)
    return output_path

def extract_dataset(dataset_path):
    with ZipFile(dataset_path) as dataset_archive:
        dataset_archive.extractall(dataset_path.parent)
