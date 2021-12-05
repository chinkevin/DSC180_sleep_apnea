import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import wfdb

def main_eda(file, outdir, **kwargs):
    
    os.makedirs(outdir, exist_ok = True)
    
    record = wfdb.rdrecord(file)
#     wfdb.plot_wfdb(record=record, title='signals', figsize=(12, 8))
    signals = record.p_signal
    sig_names = record.sig_name
    
    df = pd.DataFrame(signals, columns = sig_names)
    
    
    fig, axs = plt.subplots(3, 1, figsize=(12,6), sharey='row', sharex=True)

    axs[0].set_title('SaO2')
    axs[0].plot(df[df.SaO2 > 1].SaO2)

    axs[1].set_title('airflow')
    axs[1].plot(df[df.SaO2 > 1].AIRFLOW)

    axs[2].set_title('heartrate')
    axs[2].plot(df[df.SaO2 > 1].PR)
    plt.savefig(os.path.join(outdir, 'eda.png'))
    plt.close()