## source activate pnnl_socialsim
## python run_predictions.py --config ./metadata/configs/seed_predictions.json

import numpy as np
import pandas as pd
import sys,os
import random
import json
from ast import literal_eval
from pandas.io.json import json_normalize
import argparse
import logging
import logging.config
import json
import shutil
from os import listdir
from os.path import isfile, join
import glob
import pickle

import scipy.stats as stats
from datetime import datetime, timedelta, date

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import VotingRegressor

### import other libraries
from ml_libs.utils import *
from ml_libs.model_selection import modelSelection

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "5" # "0, 1" for multiple

class Error(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
    
models = {
    "RandomForestRegressor": RandomForestRegressor(),
    "ExtraTreesRegressor": ExtraTreesRegressor(),
    "BaggingRegressor": BaggingRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "AdaBoostRegressor": AdaBoostRegressor(),
    "GaussianProcessRegressor": GaussianProcessRegressor(),
    "IsotonicRegression": IsotonicRegression(),
    "ARDRegression": ARDRegression(),
    "HuberRegressor": HuberRegressor(),
    "LinearRegression": LinearRegression(),
    "LogisticRegression": LogisticRegression(),
    "LogisticRegressionCV": LogisticRegressionCV(),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
    "SGDRegressor": SGDRegressor(),
    "TheilSenRegressor": TheilSenRegressor(),
    "RANSACRegressor": RANSACRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "RadiusNeighborsRegressor": RadiusNeighborsRegressor(),
    "MLPRegressor": MLPRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "ExtraTreeRegressor": ExtraTreeRegressor()
}

def create_dir(x_dir):
    if not os.path.exists(x_dir):
        os.makedirs(x_dir)
        print("Created new dir. %s"%x_dir)


def reset_dir(x_dir):
    fn=create_dir(x_dir)
    
"""
Load the simulation parameters
"""
parser = argparse.ArgumentParser(description='Simulation Parameters')
parser.add_argument('--config', dest='config_file_path', type=argparse.FileType('r'))
args = parser.parse_args()

config_json=json.load(args.config_file_path)
print(config_json)

version = config_json['VERSION_TAG']
domain = config_json['DOMAIN']
platform = config_json['PLATFORM']

prediction_type = config_json['PREDICTION_TYPE']

start_train_period = config_json['start_train_period']
end_train_period = config_json['end_train_period']

start_sim_period = config_json['start_sim_period']
end_sim_period = config_json['end_sim_period']

days = config_json['time_lag']['days']
hours = config_json['time_lag']['hours']
delta = (days, hours)

global_features_path = config_json['FEATURES_PATH']['GLOBAL']
global_features_path = global_features_path.format(domain, platform) if "{0}" in global_features_path else global_features_path

local_features_path = config_json['FEATURES_PATH']['LOCAL']
local_features_path = local_features_path.format(domain, platform) if "{0}" in local_features_path else local_features_path

target_path = config_json['FEATURES_PATH']['TARGET']
target_path = target_path.format(domain, platform, prediction_type) if "{0}" in target_path else target_path

info_ids_path = config_json['INFORMATION_IDS']
info_ids_path = info_ids_path.format(domain)

ml_models = {}
ml_params= {}
test_models = config_json['MODEL_PARAMS']
for model, params in test_models.items():
    
    ml_models[model] = models[model]
    params_dict = params
    
    if "hidden_layer_sizes" in params_dict.keys():
        list_to_tuple = params_dict['hidden_layer_sizes']
        list_to_tuple = [tuple(i) for i in list_to_tuple]
        params_dict['hidden_layer_sizes'] = list_to_tuple
    ml_params[model] = params_dict

######

### Create input directories if not already created
reset_dir(local_features_path)
reset_dir(global_features_path)
reset_dir(target_path)

### Check if features and target exist

if not isinstance(local_features_path, list):
    local_features_path = glob.glob(local_features_path+'*')
else:
    local_features_path = local_features_path
    
if not isinstance(global_features_path, list):
    global_features_path = glob.glob(global_features_path+'*')
else:
    global_features_path = global_features_path
    
if len(local_features_path) == 0 and len(global_features_path) == 0:
    raise Error('Failed due to no input features...')
    
target_path = glob.glob(target_path+'*')
if len(target_path) == 0:
    raise Error('Failed due to no target...')
    
local_features=[]
global_features=[]

for feature_path in local_features_path:
    tmp=pd.read_pickle(feature_path)
    tmp=tmp.sort_index()
    local_features.append(tmp)

for feature_path in global_features_path:
    tmp=pd.read_pickle(feature_path)
    tmp=tmp.sort_index()
    global_features.append(tmp) 
    
for feature_path in target_path:
    tmp=pd.read_pickle(feature_path)
    tmp=tmp.sort_index()
    target=tmp.copy()

### Load information IDs
info_ids = pd.read_csv(info_ids_path, header=None)
info_ids.columns = ['informationID']
info_ids = sorted(list(info_ids['informationID']))
info_ids = ['informationID_'+x if 'informationID' not in x else x for x in info_ids]
print(len(info_ids),info_ids)

### define periods
start_train_period=datetime.strptime(start_train_period,"%Y-%m-%d")
# Start date must be fixed based on the time lag
start_train_period=start_train_period+timedelta(days=delta[0], hours=delta[1])
end_train_period=datetime.strptime(end_train_period,"%Y-%m-%d")

start_sim_period=datetime.strptime(start_sim_period,"%Y-%m-%d")
end_sim_period=datetime.strptime(end_sim_period,"%Y-%m-%d")

### Prepare input features and target
data_X, data_y = data_split(start_train_period, end_train_period, target, features_local=local_features,
                           features_global=global_features, delta=delta, narrative_list=info_ids)

### Train Model
train_models = modelSelection(ml_models, ml_params)

train_models.fit(data_X, data_y)

for train_model, hyper_params in train_models.best_params.items():
    print(train_model, 'best hyper parameters:', hyper_params)
    print()

### re-train best model over the whole data    
train_models.fit_all(data_X, data_y)

### Perform predictions and store results
model_results = {}
for model_id, train_model  in train_models.best_model.items():
    
    Gperformance_data, narrative_sim_data, narrative_gt_data = run_predictions(model_id, train_model, start_sim_period,
                                                                              end_sim_period, target, narrative_list=info_ids,
                                                                              features_global=global_features,
                                                                              features_local=local_features,
                                                                              delta=delta)
    
    model_results[model_id] = {'Gperformance':Gperformance_data, 'Simulations':narrative_sim_data, 'GT':narrative_gt_data}

    
### Save predictions and performance 
output_path = "./ml_output/{0}/{1}/{2}/".format(domain, platform, prediction_type)

for model_id, results in model_results.items():
    
    output_dir = "{0}_{1}_{2}_{3}".format(model_id, str(start_sim_period.strftime("%Y-%m-%d")), str(end_sim_period.strftime("%Y-%m-%d")), version)
    output_path_ = output_path+output_dir+'/'
    
    ### Create output directory
    reset_dir(output_path_)
    
    if 'Gperformance' in results.keys():
        Gperformance_data=results['Gperformance']
        Gperformance_data.to_pickle(output_path_+'Gperformance.pkl.gz')
    if 'GT' in results.keys():
        groundtruth_data = results['GT']
        pickle.dump(groundtruth_data, open(output_path_+'gt_data.pkl.gz', 'wb'))
    
    simulations_data = results['Simulations']
    pickle.dump(simulations_data, open(output_path_+'simulations_data.pkl.gz', 'wb'))
    
    best_model_params = train_models.best_params[model_id]
    pickle.dump(best_model_params, open(output_path_+'best_hyper_parameters.pkl.gz', 'wb'))
    
    best_model = train_models.best_model[model_id]
    # save the model to disk
    filename = 'best_model.h5'
    pickle.dump(best_model, open(output_path_+filename, 'wb'))
    
    print('Simulation files for {0} stored at'.format(model_id), output_path_)
    
