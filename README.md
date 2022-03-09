# DSC180 Capstone Project

Sleep apnea is a sleep disorder where breathing starts and stops intermittenly. It can cause many issues while sleeping and even increases the risk of strokes and heart attacks. Traditionally, sleep research relies on human visual scoring. However with the advancement of machine learning, sleep research can be become a highly automated process. The purpose of this respoitory is to automatically classify sleep stages specifically for people with sleep apnea. Using signals from polysomnography data, such as EEG, EMG, EOG, and ECG, we can score sleep records using a Light Gradient Boosted Machine classifier into five stages: wake state, REM, N1, N2, and N3.

### Building the project stages using `run.py`

* To get the data, from the project root dir, run `python run.py data features`
  - This fetches the data, then creates features (defined in
    `src/features.py`) and saves them in the location specified in
    `features-params.json`.
* To build a model, from the project root dir, run `python run.py data
  features model`
  - This fetches the data, creates the features, then trains a lgbm classifier
    (with parameters specified in `config`).
* To predict and validate a model, from the project root dir, run `python run.py predict validate`
  - This runs the model on validation data, analyzes its performance, and creates visualizations.
