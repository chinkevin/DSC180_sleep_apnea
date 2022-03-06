import os
import warnings
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import yasa
from preprocessing import crop_hypno, extract_features
import mne
import sleepecg

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

        # extract ECG features
        # Read ECG data
        # warnings.filterwarnings('ignore')
        raw = mne.io.read_raw(eeg_file)
        raw_data = raw.get_data()
        # you can get the metadata included in the file and a list of all channels:
        info = raw.info
        channels = raw.ch_names

        data, times = raw[:]  

        sf = 256
        heartbeat_times = sleepecg.detect_heartbeats(data[3], sf)/sf
        sleep_stage_duration = 30
        record_duration = heartbeat_times[-1]
        num_stages = record_duration // sleep_stage_duration
        # check if ECG epoch number is the same with EEG epoch number
        if features.shape[0] - num_stages > 1:
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
        fs_rri_resample = 256
        max_nans = 0.5
        feature_ids = []

        td_feature = sleepecg.feature_extraction._hrv_timedomain_features(rri,
                            rri_times,stage_times,lookback,lookforward,)
        fd_feature = sleepecg.feature_extraction._hrv_frequencydomain_features(rri,rri_times,
                    stage_times,lookback,lookforward,fs_rri_resample, max_nans, feature_ids)
    
        col = ['ECG_meanNN','ECG_maxNN','ECG_minNN','ECG_rangeNN','ECG_SDNN','ECG_RMSSD','ECG_SDSD','ECG_NN50',
       'ECG_NN20','ECG_pNN50','ECG_pNN20','ECG_medianNN','ECG_madNN','ECG_iqrNN','ECG_cvNN',
       'ECG_cvSD','ECG_meanHR','ECG_maxHR', 'ECG_minHR', 'ECG_stdHR',
       'ECG_SD1', 'ECG_SD2', 'ECG_S', 'ECG_SD1_SD2_ratio', 'ECG_CSI', 'ECG_CVI','ECG_total_power', 
       'ECG_vlf', 'ECG_lf', 'ECG_lf_norm', 'ECG_hf', 'ECG_hf_norm', 'ECG_lf_hf_ratio']
        td_feat = pd.DataFrame(td_feature)
        fd_feat = pd.DataFrame(fd_feature)
        df_ecg = pd.concat([td_feat,fd_feat],axis = 1)
        df_ecg.columns = col
        df1 = pd.DataFrame([[np.nan] * len(df_ecg.columns)], columns=df_ecg.columns)
        df_ecg = df_ecg.append(df1, ignore_index=True)
        features.reset_index(inplace=True)
        temp = pd.concat([features, df_ecg], axis=1)
        temp.set_index(['subj','epoch'])
    
    
        df.append(temp)

    df = pd.concat(df)

    # Convert to category
    df['stage'] = df['stage'].astype('str')
    # Export
    os.makedirs(out_dir, exist_ok = True)
    df.to_parquet(out_dir + "features_nsrr_shhs2.parquet")