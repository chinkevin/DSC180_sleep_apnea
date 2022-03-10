"""
Based On: https://github.com/raphaelvallat/yasa_classifier/blob/master/predict/03_predict_nsrr_shhs.ipynb
"""
import os
import yasa
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import warnings
import sleepecg

def predict(data_dir, eeg_dir, hypno_dir, out_dir, include, ecg_col):

	df_subj = pd.read_csv(data_dir)
	df_subj = df_subj.query("set == 'testing'").set_index("subj")

	df = []
	sf = 100
	models = ["eeg+eog+emg+demo", "eeg+eog+emg+ecg+demo"]

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
	        raw.drop_channels(np.setdiff1d(raw.info['ch_names'], ['ECG']+include))
	        raw.load_data()
	    except:
	        continue
	        
	    # Resample and high-pass filter 
	    raw.resample(sf, npad="auto")
	    
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
	       
	    # PREDICT SLEEP STAGES
	    md = dict(age=df_subj.loc[sub, 'age'], male=df_subj.loc[sub, 'male'])
	    # Loop across classifiers
	    for model in models:
	        path_to_model = 'output/classifiers/clf_%s_lgb_gbdt.joblib' % model
	        assert os.path.isfile(path_to_model)

	        if model == "eeg":
	            params = dict(eeg_name=include[0])
	        elif model == "eeg+demo":
	            params = dict(eeg_name=include[0], metadata=md)
	        elif model == "eeg+eog":
	            params = dict(eeg_name=include[0], eog_name=include[1])
	        elif model == "eeg+eog+demo":
	            params = dict(eeg_name=include[0], eog_name=include[1], metadata=md)
	        elif model == "eeg+eog+emg":
	            params = dict(eeg_name=include[0], eog_name=include[1], emg_name=include[2])
	        elif model == "eeg+eog+emg+demo":
	            params = dict(eeg_name=include[0], eog_name=include[1], emg_name=include[2], 
	                          metadata=md)
	        elif model == "eeg+eog+emg+ecg+demo":
	            params = dict(eeg_name=include[0], eog_name=include[1], emg_name=include[2],# ecg_name="ECG",
	                          metadata=md)

	        # Predict stages and probability

	        if model == "eeg+eog+emg+ecg+demo":
	        	# with ECG
		        sls = yasa.SleepStaging(raw, **params)
		        sls.fit()
		        sls.feature_name_ = sls.feature_name_ + ecg_col
		        clf = sls._load_model(path_to_model)
		        features =  sls.get_features()
		        
		        # extrace ecg features
	        	ecg = raw.get_data()[0] * 1e6
		        heartbeat_times = sleepecg.detect_heartbeats(ecg, sf)/sf
		        sleep_stage_duration = 30
		        # record_duration = times[-1]
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
		        X = temp.copy()[clf.feature_name_]
		        pred = clf.predict(X)
		        proba = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_)
		        proba.index.name = 'epoch'

		       
	        else:
	        	# without ECG
		        sls = yasa.SleepStaging(raw, **params)
		        proba = sls.predict_proba(path_to_model)
		        pred = sls.predict(path_to_model)


	        confidence = proba.max(1).to_numpy()
	        
	        # Get the 2nd most probable stage
	        # Using method="first" ensures that there will always be a rank 4, even
	        # when there is a tie (e.g. Wake proba is 1, zero to all others --> rank 4 = REM)
	        ranks = proba.rank(axis=1, method="first") == 4
	        hyp_pred_second = proba.columns[ranks.to_numpy().nonzero()[1]].to_numpy()

	        # Append to temporary dataframe
	        df_pred = pd.DataFrame({
	            'subj': sub,
	            'model': model,
	            'age': md['age'],
	            'male': md['male'],
	            'y_true': df_hypno.to_numpy(),
	            'y_pred': pred,#sls.predict(path_to_model),
	            'y_pred_second': hyp_pred_second,
	            'confidence': confidence,
	            'proba_N1': proba.loc[:, 'N1'].to_numpy(),
	            'proba_N2': proba.loc[:, 'N2'].to_numpy(),
	            'proba_N3': proba.loc[:, 'N3'].to_numpy(),
	            'proba_R': proba.loc[:, 'R'].to_numpy(),
	            'proba_W': proba.loc[:, 'W'].to_numpy(),
	        })

	        df.append(df_pred)

	df = pd.concat(df)

	# Remove subjects with an invalid stage
	bad_ss = df[~df['y_true'].isin(['W', 'N1', 'N2', 'N3', 'R'])]['subj'].to_numpy()
	df = df[~df['subj'].isin(bad_ss)]

	# Export to parquet, separately for each model
	for model in models:
	    if not os.path.isdir(out_dir + f"/{model}"): os.mkdir(out_dir + f"/{model}") 
	    df[df['model'] == model].to_parquet(out_dir + f"/{model}" + "/cv_loo_nsrr_shhs.parquet", index=False)