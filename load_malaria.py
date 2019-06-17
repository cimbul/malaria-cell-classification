"""
Malaria detection with cell images
==================================

This data was originally made available by NIH `here <NIH source>`_. It was posted on Kaggle by user Arunava
at `iarunava/cell-images-for-detecting-malaria`_.

Relevant excepts from the NIH website:

   This page hosts a repository of segmented cells from the thin blood smear slide images from the Malaria Screener
   research activity.

   Giemsa-stained thin blood smear slides from 150 P. falciparum-infected and 50 healthy patients were collected
   and photographed at Chittagong Medical College Hospital, Bangladesh.

   The images were manually annotated by an expert slide reader at the Mahidol-Oxford Tropical Medicine Research
   Unit in Bangkok, Thailand. 

   We applied a level-set based algorithm to detect and segment the red blood cells. The dataset contains a total
   of 27,558 cell images with equal instances of parasitized and uninfected cells.

   The CSV file for the parasitized class contains 151 patient-ID entries. The slide images for the parasitized
   patient-ID “C47P8thinOriginal” are read from two different microscope models (Olympus and Motif). The CSV file
   for the uninfected class contains 201 entries since the normal cells from the infected patients’ slides also
   make it to the normal cell category (151+50 = 201).

.. _NIH source: https://ceb.nlm.nih.gov/repositories/malaria-datasets/
.. _iarunava/cell-images-for-detecting-malaria: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
import re
from sklearn.model_selection import GroupShuffleSplit

def extract_metadata(path, infected):
    """
    Extract information from the filenames, which are encoded as `<patient>_<slide>_<cell>.png`.
    """
    name_match = re.match(r'^(.*)_IMG_(.*)_cell_(.*)\.png$', path.name)
    (patient, slide, cell) = name_match.groups()
    return {
        'infected': infected,
        'patient': patient,
        'slide': slide,
        'cell': slide + '_' + cell,
        'path': path,
    }

def load_full_dataset():
    data_dir = Path('data', 'cell_images')
    positive_dir = data_dir / 'Parasitized'
    negative_dir = data_dir / 'Uninfected'

    positive_paths = sorted(positive_dir.glob('*.png'), key = lambda p: p.name)
    negative_paths = sorted(negative_dir.glob('*.png'), key = lambda p: p.name)

    labeled_metadata_dicts = (
        [extract_metadata(p, True) for p in positive_paths] +
        [extract_metadata(p, False) for p in negative_paths])

    labeled_metadata = pd.DataFrame.from_records(
        labeled_metadata_dicts,
        columns=labeled_metadata_dicts[0].keys(), # Preserve order
    )

    return labeled_metadata

def train_test_split_by_patient(data, **kwargs):
    """
    Select a random set of patients to split into the test set.
    """
    split = GroupShuffleSplit(n_splits=1, **kwargs)
    train_indices, test_indices = next(split.split(data, groups=data['patient']))
    train_set = data.loc[train_indices].reset_index(drop=True)
    test_set = data.loc[test_indices].reset_index(drop=True)
    return train_set, test_set 

def shuffle_data(data, seed):
    shuffle_indices = np.random.RandomState(seed).permutation(len(data))
    return data.loc[shuffle_indices].reset_index(drop=True)

def load_split_dataset(seed=42):
    labeled_metadata = load_full_dataset()
    labeled_train_meta, labeled_test_meta = train_test_split_by_patient(
        labeled_metadata, random_state=seed, test_size=0.2)

    # Double-check that patients and slides were properly segregated
    assert len(set(labeled_train_meta['patient']) & set(labeled_test_meta['patient'])) == 0
    assert len(set(labeled_train_meta['slide']) & set(labeled_test_meta['slide'])) == 0

    # Pre-shuffle so we don't have to keep doing it
    labeled_train_meta = shuffle_data(labeled_train_meta, seed)
    labeled_test_meta = shuffle_data(labeled_train_meta, seed)

    return labeled_train_meta, labeled_test_meta
