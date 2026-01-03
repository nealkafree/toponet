import mne
import pyprep
import numpy as np

mne.set_log_level('WARNING')


def apply_ica(raw_data):
    """
    Applies Independent Component Analysis to raw data in order to clean artefacts.
    Divides raw EEG into independent components, drops those components that highly correlate with EOG and ECG channels
    and reconstructs raw signal from components that left.
    :return: cleaned data
    """
    # Todo: Move random state to config
    ica = mne.preprocessing.ICA(max_iter="auto", random_state=42)
    ica.fit(raw_data)
    # Threshold number of 5 is empirically tested as giving better accuracy
    # Todo: Move hyperparameter to config
    eog_indices, eog_scores = ica.find_bads_eog(raw_data, threshold=5)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_data, method='correlation', measure='zscore', threshold=5)
    ica.exclude = eog_indices + ecg_indices
    ica.apply(raw_data)
    return raw_data


def apply_prep(raw_data):
    """
    Applies PREP pipeline to raw data in order to do robust referencing and interpolate bad channels.
    PREP - https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2015.00016/full
    :return: referenced EEG signal with interpolated noisy channels
    """
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        'line_freqs': []
    }

    # Todo: Move random state to config
    prep = pyprep.PrepPipeline(raw_data, prep_params, raw_data.get_montage(), random_state=42)
    prep.fit()

    # For some reason we can't clean the list of bad channels in the object PREP returns, so copy data to a new one
    new_raw = prep.raw.copy()
    # We clean the list of bad channels so that all participants would use the same list of channels during training
    new_raw.info['bads'] = []

    return new_raw


def apply_filtering(raw_data):
    """
    Applies bandpass filter to the EEG signal.
    :return: filtered raw signal
    """
    # There were some breaks during the experient
    # In order to avoid problems we cut the experimental data to uninterrupted runs before filtering
    annotations = raw_data.annotations
    boundaries = [annotation for annotation in annotations if annotation['description'] == 'boundary']

    start = 0
    raw_segments = []
    for boundary in boundaries:
        # We extract 0.003 from the onset to avoid duplicate samples after concatenation
        # (it makes resulting segment one sample shorter)
        raw_segments.append(raw_data.copy().crop(start, boundary['onset'] - 0.003))
        start = boundary['onset']
    raw_segments.append(raw_data.copy().crop(start, raw_data.times[-1]))

    # Todo: Move frequencies into config
    for raw_segment in raw_segments:
        raw_segment.filter(l_freq=1, h_freq=40)

    # Gluing segments back together
    raw_data = mne.concatenate_raws(raw_segments)

    return raw_data


def cut_to_trials(raw_data):
    """
    Cuts the experiment to trials of one second length (starting from the onset of the stimulus)
    and applies baseline correction to them.
    :return: (list_of_trials_with_a_normal_face, list_of_trials_with_a_scrambled_face)
    """
    events = mne.events_from_annotations(raw_data)

    # We pick events based on raw labels
    # 2 - initial presentation of famous face, 10 - unfamiliar
    # 7 - initial presentation of scrambled face, 8 and 9 - immediate and delayed presentations
    face_events = mne.pick_events(events[0], include=[2, 10])
    scrambled_events = mne.pick_events(events[0], include=[7, 8, 9])

    face_epochs = mne.Epochs(raw_data.pick('eeg'), face_events, tmin=-0.3, tmax=1, preload=True)
    scrambled_epochs = mne.Epochs(raw_data.pick('eeg'), scrambled_events, tmin=-0.3, tmax=1, preload=True)

    # Data before the onset of stimulus is needed only for baseline correction, so we cut it off
    face_epochs.crop(0, 1)
    scrambled_epochs.crop(0, 1)

    # Data is saved in volts, which makes number too small, and we can't train anything
    # To combat this we multiply every data point by the multiplier of 5000
    # Todo: Move hyperparameter to a config
    voltage_multiplier = 5000
    # Converting to float32 is needed for compatibility with PyTorch
    face_trials = [np.float32(trial) * voltage_multiplier for trial in face_epochs]
    scrambled_trials = [np.float32(trial) * voltage_multiplier for trial in scrambled_epochs]

    return face_trials, scrambled_trials
