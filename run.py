#!/usr/bin/env python

import sys
import os
import json

sys.path.insert(0, 'src')

# from eda import main_eda
from data import data_setup
from features import build_features
from model import build_model
from predict import predict
from validate import validate

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    # if 'eda' in targets:
    #     with open('config/eda-params.json') as fh:
    #         data_cfg = json.load(fh)

    #     main_eda(**data_cfg)


    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        data_setup(**data_cfg)

    if 'features' in targets:
        with open('config/features-params.json') as fh:
            feats_cfg = json.load(fh)

        build_features(**feats_cfg)

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

    #     # make the data target
    #     model_build(feats, labels, **model_cfg)

    # if 'test' in targets:
    # 	with open('config/eda-params.json') as fh:
    #         eda_cfg = json.load(fh)

    #     	# make the data target
    # 		# data = get_data(**test_cfg)
    #     # move_data(**test_config)
    # 	main_eda(**eda_cfg)
    #     # convert_notebook(**eda_config)
    #     # find_metrics(**tuning_config)
    #     # create_launch_files(**generate_config)

    return

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)