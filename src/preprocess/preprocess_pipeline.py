import os
import random

import mne

import src.preprocess.preprocess_functions as preprocess

mne.set_log_level('WARNING')

# Todo: Move path to config
ORIGINAL_DATA_DIRECTORY = './topography/data/FaceRecognition'


def load_subject_data(subject_name):
    file_path = os.path.join(ORIGINAL_DATA_DIRECTORY, f"{subject_name}/eeg/{subject_name}_task-FaceRecognition_eeg.set")
    raw_data = mne.io.read_raw_eeglab(file_path, preload=True)

    # We need to manually set eog and ecg channels, because mne does not recognise them in this case
    raw_data.set_channel_types({'EEG063': 'ecg', 'EEG064': 'ecg', 'EEG061': 'eog', 'EEG062': 'eog'})
    return raw_data


def preprocess_data(raw_data):
    """
    Filtering, cleaning,  cutting raw signal to trials and dividing to test and train sets.
    :return: (train_set, test_set) of (trial, label) as items.
    """
    # Applying bandpass filter
    raw_data = preprocess.apply_filtering(raw_data)

    # Applying PREP pipeline in order to robustly reference signal and interpolate bad channels
    raw_data = preprocess.apply_prep(raw_data)

    # Applying ICA to clean ocular and heartbeat artefacts
    raw_data = preprocess.apply_ica(raw_data)

    # Cutting experiment to trials of one second length with baseline correction
    # We get two sets for two conditions
    face_trials, scrambled_trials = preprocess.cut_to_trials(raw_data)

    # Adding labels to trials
    face_trials = [(trial, 0) for trial in face_trials]
    scrambled_trials = [(trial, 1) for trial in scrambled_trials]

    # Splitting to train and test sets
    face_train, face_test = get_train_test_split(face_trials, 0.1)
    scrambled_train, scrambled_test = get_train_test_split(scrambled_trials, 0.1)

    return face_train + scrambled_train, face_test + scrambled_test


def get_train_test_split(trials, test_fraction):
    """
    Quick implementation of a split, just to not import additional libraries.
    Randomly splits dataset to test set of the test_fraction*len(trials) size and the train set (the rest).
    :return: (train_set, test_set)
    """
    # Todo: Move random seed to config
    random.Random(42).shuffle(trials)
    test_size = int(len(trials) * test_fraction)
    return trials[test_size:], trials[:test_size]
