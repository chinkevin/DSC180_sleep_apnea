import os, glob
import numpy as np
import pandas as pd

def data_setup(out_dir, desc_dir, usecols):
    # Define root data paths, where the NSRR data is stored
    # root_dir = '/data'
    # assert os.path.isdir(root_dir)

    # Define the output path
    # out_dir = 'output/split/'
    os.makedirs(out_dir, exist_ok = True)
    # desc_dir = 'data/datasets/shhs2-dataset-0.17.0.csv'
    # usecols = ['nsrrid', 'visitnumber', 'gender', 'age_s1', 'overall_shhs2', 
    #            'race', 'bmi_s2', 'ahi_a0h3', 'htnderv_s2'] # diabetes is only in shh1?

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

    # Keep only "Excellent" quality study
    # print(df_shhs[df_shhs['overall'] < 6].shape[0], 
    #       'subjects with bad PSG data quality will be removed.')
    # df_shhs = df_shhs[df_shhs['overall'] >= 6]

    df_shhs['male'] = (df_shhs['gender'] == 1).astype(int)

    # Keep only first visit
    df_shhs = df_shhs[df_shhs['visitnumber'] == 2]

    # Convert to str
    df_shhs['subj'] = df_shhs['subj'].apply(lambda x: str(x).zfill(4))
    df_shhs.set_index('subj', inplace=True)

    # # Define training / testing
    # # Keep only a random subset of 600 subjects for training to avoid dataset imbalance
    df_shhs["set"] = "excluded"
    idx_train = df_shhs.sample(n=600, replace=False, random_state=42).index
    idx_test = np.setdiff1d(df_shhs.index, idx_train)
    # Now we keep 100 random participants of ``idx_test`` for testing
    rs = np.random.RandomState(42)
    idx_test = rs.choice(idx_test, size=100, replace=False)
    df_shhs.loc[idx_train, "set"] = "training"
    df_shhs.loc[idx_test, "set"] = "testing"


    # ---- Test with 3 ----
    # df_shhs["set"] = "excluded"
    # idx_train = ['200077', '200078']
    # idx_test = ['200079']
    # # Now we keep 100 random participants of ``idx_test`` for testing
    # rs = np.random.RandomState(42)
    # df_shhs.loc[idx_train, "set"] = "training"
    # df_shhs.loc[idx_test, "set"] = "testing"
    # -------- end --------

    # Export demographics to CSV file
    # df_shhs['dataset'] = 'SHHS'
    # df_shhs.to_csv(out_dir + "demo_nsrr_shhs.csv")

    # print(df_shhs.shape[0], 'subjects remaining')
    # print(df_shhs['set'].value_counts())
    # df_shhs.head(10)


    df_shhs.drop(columns=['gender', 'visitnumber'], inplace=True)
    # df_shhs = df_shhs.set_index("dataset", append=True).reorder_levels(["dataset", "subj"])

    # Remove "excluded"
    # df_shhs = df_shhs[df_shhs["set"] != "excluded"]

    df_shhs['hypertension'] = df_shhs['hypertension'].astype(float)
    df_shhs['hypertension'].value_counts(dropna=False)

    # Re-order columns
    cols_order = [
      'age', 'male', 'bmi', 'ahi', 'ethnicity', 'set', 
      'hypertension']
    df_shhs = df_shhs[cols_order]

    # Export to .csv
    df_shhs.to_csv(out_dir + "shhs_demo.csv", index=True)