
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import pywt

def main_eda(file, outdir, signals, events, colors, **kwargs):
    
    os.makedirs(outdir, exist_ok = True)
    
    # read file
    record = wfdb.rdrecord(file)
    ann = wfdb.rdann(file, extension='resp')
    df = pd.DataFrame(record.p_signal, columns = record.sig_name)
    units = dict(zip(record.sig_name, record.units))
    notes = pd.Series(ann.aux_note).str.split().to_list()
    resp = pd.DataFrame(notes, ann.sample, columns=['type', 'duration', '% decrease', 'min SaO2'])
    
    # clean data
    df.SaO2[df.SaO2<10] = np.nan
    df.SaO2 = df.SaO2.interpolate()
    df.PR[df.PR<10] = np.nan
    df.PR = df.PR.interpolate()
    
    event_colors = dict(zip(events, colors))

    fig, axs = plt.subplots(len(signals)+1, 1, figsize=(15,12), sharey='row', sharex=True)

    # plot signals
    for i, signal in enumerate(signals):
        axs[i].set_title(signal)
        axs[i].plot(df[signal])
        axs[i].set_ylabel(units[signal])


    # plot events
    for event in events:
        plt.plot(resp[resp.type == event].index, np.zeros(resp[resp.type == event].shape[0]), event_colors[event])
    plt.title('events')
    plt.legend(events)
    plt.yticks([])
    plt.xlabel('seconds')
    
    plt.savefig(os.path.join(outdir, 'eda.png'))
    plt.close()

    sample_len = 300 # could add this to config files
    
    cwtmatr, freqs = pywt.cwt(
        df.ECG.values[:+sample_len], 
        scales=np.arange(1, sample_len+1), 
        wavelet = 'morl',
        method= 'conv'
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(cwtmatr, extent=[0, sample_len, 1, sample_len+1], cmap='bwr', aspect='auto',
                vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

    plt.title('ECG cwt')
    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(outdir, 'ECG_cwt.png'))
    plt.close()


# def main_eda(file, outdir, **kwargs):

#     os.makedirs(outdir, exist_ok = True)
    
#     record = wfdb.rdrecord(file)
# #     wfdb.plot_wfdb(record=record, title='signals', figsize=(12, 8))
#     signals = record.p_signal
#     sig_names = record.sig_name
    
#     df = pd.DataFrame(signals, columns = sig_names)
    
    
#     fig, axs = plt.subplots(3, 1, figsize=(12,6), sharey='row', sharex=True)

#     axs[0].set_title('SaO2')
#     axs[0].plot(df[df.SaO2 > 1].SaO2)

#     axs[1].set_title('airflow')
#     axs[1].plot(df[df.SaO2 > 1].AIRFLOW)

#     axs[2].set_title('heartrate')
#     axs[2].plot(df[df.SaO2 > 1].PR)
#     plt.savefig(os.path.join(outdir, 'eda.png'))
#     plt.close()