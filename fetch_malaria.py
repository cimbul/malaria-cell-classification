#!/usr/bin/env python

from kaggle import fetch_kaggle_dataset, extract_dataset

def fetch_malaria():
    dataset_archive_path = fetch_kaggle_dataset('iarunava/cell-images-for-detecting-malaria', 'cell_images.zip')
    extract_dataset(dataset_archive_path)

if __name__ == '__main__':
    fetch_malaria()
