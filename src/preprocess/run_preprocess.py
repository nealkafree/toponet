"""
Preprocesses data for all subjects from the original dataset and stores them in "preprocessed_data" directory.
"""

import os
import pickle
import sys

from tqdm import tqdm

import preprocess_pipeline

if __name__ == '__main__':

    PREPROCESSED_DATA_DIRECTORY = './preprocessed_data'
    original_data = sys.argv[1]

    d_list = os.listdir(original_data)

    for d in tqdm(d_list):
        if d.startswith('sub'):
            raw_data = preprocess_pipeline.load_subject_data(d, original_data)
            train_set, test_set = preprocess_pipeline.preprocess_data(raw_data)

            os.makedirs(os.path.join(PREPROCESSED_DATA_DIRECTORY, d), exist_ok=True)
            # Data is not big, so we just use pickle
            with open(os.path.join(PREPROCESSED_DATA_DIRECTORY, d, 'train.pkl'), "wb") as file:
                pickle.dump(train_set, file)
            with open(os.path.join(PREPROCESSED_DATA_DIRECTORY, d, 'test.pkl'), "wb") as file:
                pickle.dump(test_set, file)