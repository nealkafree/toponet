import os
import random

import mne

from . import preprocess_functions

mne.set_log_level('CRITICAL')

def load_subject_data(subject_name: str, original_data: str) -> mne.io.Raw:
    file_path = os.path.join(original_data, f"{subject_name}/eeg/{subject_name}_task-FaceRecognition_eeg.set")
    raw_data = mne.io.read_raw_eeglab(file_path, preload=True)

    # We need to manually set eog and ecg channels, because mne does not recognise them in this case
    raw_data.set_channel_types({'EEG063': 'ecg', 'EEG064': 'ecg', 'EEG061': 'eog', 'EEG062': 'eog'})
    return raw_data


def preprocess_data(raw_data: mne.io.Raw) -> (list, list):
    """
    Filtering, cleaning,  cutting raw signal to trials and dividing to test and train sets.
    :return: (train_set, test_set) of (trial, label) as items.
    """
    # Applying bandpass filter
    raw_data = preprocess_functions.apply_filtering(raw_data)

    # Applying PREP pipeline in order to robustly reference signal and interpolate bad channels
    raw_data = preprocess_functions.apply_prep(raw_data)

    # Applying ICA to clean ocular and heartbeat artefacts
    raw_data = preprocess_functions.apply_ica(raw_data)

    # Cutting experiment to trials of one second length with baseline correction
    # We get two sets for two conditions
    face_trials, scrambled_trials = preprocess_functions.cut_to_trials(raw_data)

    # Adding labels to trials
    face_trials = [(trial, 0) for trial in face_trials]
    scrambled_trials = [(trial, 1) for trial in scrambled_trials]

    # Splitting to train and test sets
    face_train, face_test = get_train_test_split(face_trials, 0.1)
    scrambled_train, scrambled_test = get_train_test_split(scrambled_trials, 0.1)

    return face_train + scrambled_train, face_test + scrambled_test


def get_train_test_split(trials: list, test_fraction: float) -> (list, list):
    """
    Quick implementation of a split, just to not import additional libraries.
    Randomly splits dataset to test set of the test_fraction*len(trials) size and the train set (the rest).
    :return: (train_set, test_set)
    """
    random.Random(42).shuffle(trials)
    test_size = int(len(trials) * test_fraction)
    return trials[test_size:], trials[:test_size]
