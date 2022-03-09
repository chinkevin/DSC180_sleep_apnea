import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import sklearn.metrics as skm
from helper_functions import NUM2STR, STR2NUM
from helper_functions import mean_std, median_iqr, perc_transition

def validate(data_dir, eeg_dir, hypno_dir, out_dir, w_dir, dic_features, feat_fp, ecg_feat_fp, ecg_col):

    # Choose model (set in config) -------------------
    models = ["eeg+eog+emg+demo", "eeg+eog+emg+ecg+demo"]
    # model = "eeg+eog+emg+ecg+demo"
    # model = "eeg+eog+emg+demo"

    for model in models:

        df = pd.read_parquet(w_dir + model + "/cv_loo_nsrr_shhs.parquet")
        df['subj'] = df['subj'].astype(str)
        df_demo = pd.read_csv(data_dir)


        # Per night
        df_scores = []

        labels = ['N1', 'N2', 'N3', 'R', 'W']

        for sub in df['subj'].unique():
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

        # -------- Plots --------

        # Confusion Matrix
        cm = 100 * skm.confusion_matrix(df['y_true'], df['y_pred'], labels=labels, normalize="true")
        cm = pd.DataFrame(cm, index=labels, columns=labels)

        # Plot
        fig, ax = plt.subplots(1, 1, dpi=100, figsize=(4.5, 4.5))
        sns.heatmap(cm, annot=True, vmin=0, vmax=100, cmap="Blues", square=True, cbar=False, fmt=".1f")
        plt.ylabel("Reference")
        plt.xlabel("Predicted")
        plt.title(f"{model} Confusion Matrix", y=1.02)
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok = True)
        plt.savefig(out_dir + f"{model}_cv_confusion_matrix.png", dpi=300, bbox_inches="tight")


        # F1 Score
        cmap_stages = ['#99d7f1', '#009DDC', 'xkcd:twilight blue', 'xkcd:rich purple', 'xkcd:sunflower']
        df_f1 = df_scores[['f1_N1', 'f1_N2', 'f1_N3', 'f1_R', 'f1_W']].copy()
        df_f1.columns = df_f1.columns.str.split('_').str.get(1)

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=100)
        sns.boxplot(data=df_f1, palette=cmap_stages, fliersize=0, ax=ax, saturation=1, notch=True)
        plt.title(f"{model} F1 Scores", y=1.02)
        plt.xlabel("Stage")
        plt.ylabel("F1-score")
        sns.despine()
        plt.savefig(out_dir + f"{model}_f1_scores.png", dpi=300, bbox_inches="tight")
        


        # Feature Importance
        
        fimp = 'output/classifiers/clf_%s_lgb_gbdt.csv' % model
        df_fimp = pd.read_csv(fimp).head(20)
        df_fimp['Features'] = df_fimp['Features'].replace(dic_features)

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        sns.barplot(data=df_fimp, y="Features", x="Importance", palette="magma", saturation=1)

        plt.ylabel("")
        plt.xlabel("Importance (SHAP values)")
        plt.title(f"{model} Feature Importance", y=1.02)
        sns.despine()
        plt.tight_layout()
        plt.savefig(out_dir + f"{model}_cv_fimp.png", dpi=300, bbox_inches="tight")


        # Feature Correlation

        # ecg feature correlation
        if model == "eeg+eog+emg+ecg+demo":
            df = pd.read_parquet(ecg_feat_fp)
            plt.figure(figsize=(12,8))
            plt.title("ECG Feature Correlation", y=1.02)
            sns.heatmap(df[ecg_col].corr(), cmap="RdBu")
            plt.savefig(out_dir + "ecg_feature_corr.png", dpi=300, bbox_inches="tight")
        # remaining feature correlation
        else:
            df = pd.read_parquet(feat_fp)
            plt.figure(figsize=(12,8))
            plt.title("Feature Correlation", y=1.02)
            sns.heatmap(df.corr(), cmap="RdBu")
            plt.savefig(out_dir + "feature_corr.png", dpi=300, bbox_inches="tight")

