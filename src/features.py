import os
import warnings
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import yasa
from preprocessing import crop_hypno, extract_features

def build_features(data_dir, eeg_dir, hypno_dir, out_dir, include):
    # Define paths (can be defined in config files)
    # eeg_dir = 'data/edfs/shhs2/'
    # hypno_dir = 'data/annotations-events-profusion/shhs2/'
    # # parent_dir = os.path.dirname(os.getcwd())
    # out_dir = '/output/features/'

    df_subj = pd.read_csv(data_dir)
    df_subj = df_subj.query("set == 'training'").set_index("subj")

    # print(df_subj.shape[0], 'subjects remaining')

    df = []
    # include = ['EEG', 'EOG(L)', 'EMG']
    sf = 100

    for sub in df_subj.index:
        eeg_file = eeg_dir + 'shhs2-' + str(sub) + '.edf'
        hypno_file = hypno_dir + 'shhs2-' + str(sub) + '-profusion.xml'

        # Check that file exists
        if not os.path.isfile(eeg_file):
            warnings.warn("File not found %s" % eeg_file)
            continue
        if not os.path.isfile(hypno_file):
            warnings.warn("File not found %s" % hypno_file)
            continue

        # LOAD EEG DATA
        try:
            raw = read_raw_edf(eeg_file, preload=False, verbose=0)
            raw.drop_channels(np.setdiff1d(raw.info['ch_names'], include))
            # Skip subjects if channel were not found
            assert len(raw.ch_names) == len(include)
            raw.load_data()
        except:
            continue
        
        # Resample and high-pass filter 
        raw.resample(sf, npad="auto")
        
        # LOAD HYPNOGRAM
        hypno, sf_hyp = yasa.load_profusion_hypno(hypno_file)
        # (Optional) We keep up to 15 minutes before / after sleep
        # hypno, tmin, tmax = crop_hypno(hypno)
        # raw.crop(tmin, tmax)
        # Check that hypno and data have the same number of epochs
        n_epochs = hypno.shape[0]
        if n_epochs != np.floor(raw.n_times / sf / 30):
            print("- Hypno and data size do not match.")
            continue
        
        # Convert hypnogram to str
        df_hypno = pd.Series(hypno)
        df_hypno.replace({0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}, inplace=True)
        stage_min = df_hypno.value_counts(sort=False) / 2

        # INCLUSION CRITERIA (DISABLED)
        # Hypnogram must include all stages
    #     if np.unique(hypno).tolist() != [0, 1, 2, 3, 4]:
    #         print("- Not all stages are present.")
    #         continue
    #     # If the duration is not between 4 to 12 hours, skip subject
    #     if not(4 < n_epochs / 120 < 12):
    #         print("- Recording too short/long.")
    #         continue

        # EXTRACT FEATURES
        features = extract_features(df_subj, sub, raw, include)
        # Add hypnogram
        features['stage'] = df_hypno.to_numpy()
        df.append(features)

    df = pd.concat(df)

    # Convert to category
    df['stage'] = df['stage'].astype('str')
    # Export
    os.makedirs(out_dir, exist_ok = True)
    df.to_parquet(out_dir + "features_nsrr_shhs2.parquet")