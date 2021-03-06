"""
Based On: https://github.com/raphaelvallat/yasa_classifier/blob/master/feature_extraction/01_features_nsrr_shhs.ipynb
"""
import os
import warnings
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import yasa
from preprocessing import crop_hypno, extract_features
import mne
import sleepecg

def build_features_ecg(data_dir, eeg_dir, hypno_dir, out_dir, include, ecg_col):

    df_subj = pd.read_csv(data_dir)
    df_subj = df_subj.query("set == 'training'").set_index("subj")

    df = []
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
            raw.drop_channels(np.setdiff1d(raw.info['ch_names'], ["ECG"] + include))
            # Skip subjects if channel were not found
            assert len(raw.ch_names) == 1+len(include)
            raw.load_data()
        except:
            continue
        
        # Resample and high-pass filter 
        raw.resample(sf, npad="auto")
        ecg = raw.get_data()[0]
        
        # LOAD HYPNOGRAM
        hypno, sf_hyp = yasa.load_profusion_hypno(hypno_file)
        # Check that hypno and data have the same number of epochs
        n_epochs = hypno.shape[0]
        if n_epochs != np.floor(raw.n_times / sf / 30):
            print("- Hypno and data size do not match.")
            continue
        
        # Convert hypnogram to str
        df_hypno = pd.Series(hypno)
        df_hypno.replace({0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}, inplace=True)
        stage_min = df_hypno.value_counts(sort=False) / 2

        # EXTRACT FEATURES
        features = extract_features(df_subj, sub, raw, include)
        # Add hypnogram
        features['stage'] = df_hypno.to_numpy()

        # extract ECG features
        heartbeat_times = sleepecg.detect_heartbeats(ecg, sf)/sf
        sleep_stage_duration = 30
        num_stages = features.shape[0]
        # check if ECG epoch number is the same with EEG epoch number
        if (features.shape[0] - num_stages) != 0:
            print('Skip due to different numbers of epoch')
            continue
        stage_times = np.arange(num_stages) * sleep_stage_duration
        min_rri, max_rri = None, None
        lookback, lookforward = 240, 60
        rri = sleepecg.preprocess_rri(
                np.diff(heartbeat_times),
                min_rri=min_rri,
                max_rri=max_rri,
            )
        rri_times = heartbeat_times[1:]
        
        fs_rri_resample = 100
        max_nans = 0.5
        feature_ids = []

    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            td_feature = sleepecg.feature_extraction._hrv_timedomain_features(rri,
                            rri_times,stage_times,lookback,lookforward,)
            fd_feature = sleepecg.feature_extraction._hrv_frequencydomain_features(rri,rri_times,
                        stage_times,lookback,lookforward,fs_rri_resample, max_nans, feature_ids)

        td_feat = pd.DataFrame(td_feature)
        fd_feat = pd.DataFrame(fd_feature)
        df_ecg = pd.concat([td_feat,fd_feat],axis = 1)
        df_ecg.columns = ecg_col
        features.reset_index(inplace=True)
        temp = pd.concat([features, df_ecg], axis=1)
        temp.set_index(['subj','epoch'])


        df.append(temp)

    df = pd.concat(df).set_index(['subj','epoch'])

    # Convert to string
    df['stage'] = df['stage'].astype('str')
    # Export
    os.makedirs(out_dir, exist_ok = True)
    df.to_parquet(out_dir + "features_ecg_nsrr_shhs2.parquet")