#!/usr/bin/env python

import sys
import os
import json

sys.path.insert(0, 'src')

from data import data_setup
from features import build_features
from features_with_ECG import build_features_ecg
from model import build_model
from predict import predict
from validate import validate

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    '''

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        data_setup(**data_cfg)

    if 'features' in targets:
        with open('config/features-params.json') as fh:
            feats_cfg = json.load(fh)

        build_features(**feats_cfg)

    if 'features_ecg' in targets:
        with open('config/features-params.json') as fh:
            feats_cfg = json.load(fh)

        build_features_ecg(**feats_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        build_model(**model_cfg)

    if 'predict' in targets:
        with open('config/predict-params.json') as fh:
            predict_cfg = json.load(fh)

        predict(**predict_cfg)

    if 'validate' in targets:
        with open('config/validate-params.json') as fh:
            validate_cfg = json.load(fh)

        validate(**validate_cfg)

    return

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)