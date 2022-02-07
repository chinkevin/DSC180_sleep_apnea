import os
import pandas as pd
import joblib
from lightgbm import LGBMClassifier

# Define hyper-parameters
params = dict(
    boosting_type='gbdt',
    n_estimators=400,
    max_depth=5,
    num_leaves=90,
    colsample_bytree=0.5,
    importance_type='gain',
    class_weight=None
)

out_dir = "output/classifiers/"
os.makedirs(out_dir, exist_ok = True)

df = pd.read_parquet("output/features/features_nsrr_shhs2.parquet")
df_demo = pd.read_csv("output/demographics/shhs_demo.csv")
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
}

# Define target and groups
y = df['stage']
subjects = df.index.get_level_values(0).to_numpy()

# Export a full list of features
features = pd.Series(X_all['eeg+eog+emg+demo'].columns, name="Features")
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
        fname = out_dir + 'clf_%s_lgb_%s_%s.joblib' %         (name, params['boosting_type'], class_weight)
    else:
        fname = out_dir + 'clf_%s_lgb_%s.joblib' %         (name, params['boosting_type'])
        
    # Export model
    joblib.dump(clf, fname, compress=True)
    
    # Also save directly to YASA
    # out_dir_yasa = "/Users/raphael/GitHub/yasa/yasa/classifiers/"
    # fname_yasa = out_dir_yasa + 'clf_%s_lgb.joblib' % name
    # joblib.dump(clf, fname_yasa, compress=True)
    
    # Features importance (full model only)
    if name == "eeg+eog+emg+demo":
        # Export LGBM feature importance
        df_imp = pd.Series(clf.feature_importances_, index=clf.feature_name_, name='Importance').round()
        df_imp.sort_values(ascending=False, inplace=True)
        df_imp.index.name = 'Features'
        df_imp.to_csv(fname[:-7] + ".csv")