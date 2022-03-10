"""
Based On: https://github.com/raphaelvallat/yasa_classifier/blob/master/02_train_export_classifier.ipynb
"""
import os
import pandas as pd
import joblib
from lightgbm import LGBMClassifier


def build_model(data_dir, eeg_dir, hypno_dir, out_dir, feat_fp, include, ecg_col):
    # Define hyper-parameters (can add to config)
    params = dict(
        boosting_type='gbdt',
        n_estimators=400,
        max_depth=5,
        num_leaves=90,
        colsample_bytree=0.5,
        importance_type='gain',
        class_weight=None
    )

    # out_dir = "output/classifiers/"
    os.makedirs(out_dir, exist_ok = True)

    df = pd.read_parquet(feat_fp)
    df_demo = pd.read_csv(data_dir)
    # Remove columns that are already present in `df`
    df_demo.drop(columns=['male', 'age'], inplace=True)

    # different predictors for different column groupings
    cols_all = df.columns
    cols_time = cols_all[cols_all.str.startswith('time_')].tolist()
    # EEG also includes the time columns
    cols_eeg = cols_all[cols_all.str.startswith('eeg_')].tolist() + cols_time  
    cols_eog = cols_all[cols_all.str.startswith('eog_')].tolist()
    cols_emg = cols_all[cols_all.str.startswith('emg_')].tolist()
    cols_demo = ['age', 'male']

    # Define predictors
    X_all = {
        'eeg': df[cols_eeg],
        'eeg+demo': df[cols_eeg + cols_demo],
        
        'eeg+eog': df[cols_eeg + cols_eog],
        'eeg+eog+demo': df[cols_eeg + cols_eog + cols_demo],
        
        'eeg+emg': df[cols_eeg + cols_emg],
        'eeg+emg+demo': df[cols_eeg + cols_emg + cols_demo],
        
        'eeg+eog+emg': df[cols_eeg + cols_eog + cols_emg],
        'eeg+eog+emg+demo': df[cols_eeg + cols_eog + cols_emg + cols_demo],

        'eeg+eog+emg+ecg+demo': df[cols_eeg + cols_eog + cols_emg + ecg_col + cols_demo],
    }

    # Define target and groups
    y = df['stage']
    subjects = df.index.get_level_values(0).to_numpy()

    # Export a full list of features
    features = pd.Series(X_all['eeg+eog+emg+ecg+demo'].columns, name="Features")
    features.to_csv("features.csv", index=False)

    # Parallel processing when building the trees.
    params['n_jobs'] = 8

    # Loop across combs of predictors
    for name, X in X_all.items():
        
        # Fit
        clf = LGBMClassifier(**params)
        clf.fit(X, y)

        # Print the accuracy on the training dataset: shouldn't be too high..!
        print("%s (%i features) - training accuracy: %.3f" % 
            (name, X.shape[1], clf.score(X, y)))
        
        # Export trained classifier
        if params['class_weight'] is not None:
            fname = out_dir + 'clf_%s_lgb_%s_%s.joblib' % (name, params['boosting_type'], class_weight)
        else:
            fname = out_dir + 'clf_%s_lgb_%s.joblib' % (name, params['boosting_type'])
            
        # Export model
        joblib.dump(clf, fname, compress=True)
        
        # Features importance (full models only)
        if name == "eeg+eog+emg+demo" or name == 'eeg+eog+emg+ecg+demo':
            # Export LGBM feature importance
            df_imp = pd.Series(clf.feature_importances_, index=clf.feature_name_, name='Importance').round()
            df_imp.sort_values(ascending=False, inplace=True)
            df_imp.index.name = 'Features'
            df_imp.to_csv(fname[:-7] + ".csv")