"""
Based On: https://github.com/raphaelvallat/yasa_classifier/blob/master/00_randomize_train_test.ipynb
"""
import os, glob
import numpy as np
import pandas as pd

def data_setup(out_dir, desc_dir, usecols):
    
    os.makedirs(out_dir, exist_ok = True)

    df_shhs = pd.read_csv(desc_dir, usecols=usecols, encoding='cp1252') # enconding to handle special characters

    # Rename columns
    df_shhs.rename(columns={'nsrrid': 'subj',
                          'age_s1': 'age',
                          'overall_shhs2': 'overall',
                          'bmi_s2': 'bmi',
                          'ahi_a0h3': 'ahi',
                          'htnderv_s2': 'hypertension',
                        }, inplace=True)

    df_shhs['race'].replace({1: 'caucasian', 2: 'african', 3: 'other'}, inplace=True)
    df_shhs.loc[df_shhs['race'] == 1, 'race'] = 'hispanic'
    df_shhs.rename(columns={'race': 'ethnicity'}, inplace=True)

    df_shhs['male'] = (df_shhs['gender'] == 1).astype(int)

    # Keep only first visit
    df_shhs = df_shhs[df_shhs['visitnumber'] == 2]

    # Convert to str
    df_shhs['subj'] = df_shhs['subj'].apply(lambda x: str(x).zfill(4))
    df_shhs.set_index('subj', inplace=True)

    # # Define training / testing
    # # Keep only a random subset of 600 subjects for training to speed up training
    df_shhs["set"] = "excluded"
    idx_train = df_shhs.sample(n=600, replace=False, random_state=42).index
    idx_test = np.setdiff1d(df_shhs.index, idx_train)
    # Now we keep 100 random participants of ``idx_test`` for testing
    rs = np.random.RandomState(42)
    idx_test = rs.choice(idx_test, size=100, replace=False)
    df_shhs.loc[idx_train, "set"] = "training"
    df_shhs.loc[idx_test, "set"] = "testing"

    df_shhs.drop(columns=['gender', 'visitnumber'], inplace=True)

    df_shhs['hypertension'] = df_shhs['hypertension'].astype(float)
    df_shhs['hypertension'].value_counts(dropna=False)

    # Re-order columns
    cols_order = [
      'age', 'male', 'bmi', 'ahi', 'ethnicity', 'set', 
      'hypertension']
    df_shhs = df_shhs[cols_order]

    # Export to .csv
    df_shhs.to_csv(out_dir + "shhs_demo.csv", index=True)