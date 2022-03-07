import os
import glob
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import sklearn.metrics as skm
from helper_functions import NUM2STR, STR2NUM
from helper_functions import mean_std, median_iqr, perc_transition

def validate(data_dir, eeg_dir, hypno_dir, out_dir, w_dir):

    # Choose model (set in config) -------------------
    model = "eeg+eog+emg+ecg+demo"

    df = pd.read_parquet(w_dir + "/cv_loo_nsrr_shhs.parquet")
    df['subj'] = df['subj'].astype(str)
    df_demo = pd.read_csv(data_dir)


    # Per each night
    df_scores = []

    labels = ['N1', 'N2', 'N3', 'R', 'W']

    for sub in tqdm(df['subj'].unique(), leave=False):
        df_sub = df[df['subj'] == sub]
        yt = df_sub['y_true']
        yp = df_sub['y_pred']
        n = yt.shape[0] 

        sub_scores = {
            # Accuracy
            'accuracy': 100 * skm.accuracy_score(yt, yp),
            'kappa': 100 * skm.cohen_kappa_score(yt, yp, labels=labels),
            'mcc': 100 * skm.matthews_corrcoef(yt, yp),
            'f1_macro': 100 * skm.f1_score(yt, yp, labels=labels, average='macro', zero_division=1),
            # % Transitions
            'dur_hours': (yt.size / 2) / 60,
            'perc_trans_true': perc_transition(yt),
            'perc_trans_pred': perc_transition(yp),
            # Confidence
            'avg_confidence': 100 * df_sub['confidence'].mean()
        }

        # F1 for each stage
        f1 = 100 * skm.f1_score(yt, yp, average=None, labels=labels, zero_division=1)
        for f, l in zip(f1, labels):
            sub_scores['f1_' + l] = f
            
        # In the case of an error, is the second-highest probability typically the correct one?
        error = (yp != yt).to_numpy()
        sub_scores['accuracy_second'] = 100 * skm.accuracy_score(yt[error], df_sub['y_pred_second'][error])

        # Proportion of each stage (NaN = 0)
        prop_true = (yt.value_counts() / n).add_prefix('perc_').add_suffix('_true')
        prop_pred = (yp.value_counts() / n).add_prefix('perc_').add_suffix('_pred')
        sub_scores.update(prop_true.to_dict())
        sub_scores.update(prop_pred.to_dict())

        # Append to main dataframe
        df_scores.append(pd.DataFrame(sub_scores, index=[sub]))

    df_scores = pd.concat(df_scores)
    df_scores.sort_index(axis=1, inplace=True)
    df_scores.index.name = 'subj'

    # Join with demographics
    df_scores = df_scores.join(df_demo, how="left")

    os.makedirs(w_dir + model, exist_ok = True)
    # export to csv
    df_scores.round(3).to_csv(w_dir + model + "/df_scores.csv")