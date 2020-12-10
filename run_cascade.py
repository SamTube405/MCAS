## source activate pnnl_socialsim
## python run_cascade.py --config ./metadata/configs/twitter_cve_S2.json
import numpy as np
import pandas as pd
import sys,os
# from random import choices
import random
from datetime import datetime as dt
import json
from ast import literal_eval
from pandas.io.json import json_normalize

import argparse
import logging
import logging.config
import json
import shutil
import os
import shutil
from os import listdir
from os.path import isfile, join

from libs.lib_cascade_inf import *
from joblib import Parallel, delayed

NUMBER_OF_SEED_CHUNKS=100

# def del_dir(x_dir):
#     if(os.path.exists(x_dir)==True):
#         try:
#             shutil.rmtree(x_dir)
#             print("Deleted. %s"%x_dir)
#         except OSError as e:
#             print("Error: %s - %s." % (e.filename, e.strerror))
            
def create_dir(x_dir):
    if not os.path.exists(x_dir):
        os.makedirs(x_dir)
        print("Created new dir. %s"%x_dir)


def reset_dir(x_dir):
    create_dir(x_dir)

# def _get_input_posts(path):
#     input_posts_lines=[]
#     json_file=open(path,"r")
#     for line in json_file.readlines():
#         doc=json.loads(line)
#         doc = {
#                 "id_h":doc["nodeID"],
#                 "author_h":doc["nodeUserID"],
#                 "created_date":doc["nodeTime"],
#                 "informationID":doc["informationID"],
#                 }
        
#         input_posts_lines.append(doc)
        
#     input_posts_records=json_normalize(input_posts_lines)
#     input_posts_records.set_index(pd.DatetimeIndex(input_posts_records['created_date']),inplace=True)
#     return input_posts_records

def _get_input_messages(path):
    input_messages=pd.read_csv(path,header=None)
    input_messages.columns=['nodeTime','nodeID','nodeUserID','informationID']
    ##input_messages.set_index(pd.DatetimeIndex(input_messages['nodeTime']),inplace=True)
    infoIDs=input_messages['informationID'].unique()
    print("Presence: ",infoIDs.shape[0],infoIDs)
    ##input_messages=input_messages.iloc[0:100]
    
#     input_messages_array=[]
#     for infoID in infoIDs:
#         input_messages_=input_messages.query('informationID==@infoID')
#         input_messages_array.append(input_messages_)
    input_messages_array=np.array_split(input_messages, NUMBER_OF_SEED_CHUNKS)
    return input_messages_array


def _run(args):
    iposts_records=args[0]
    index=args[1]
    version="%s-%d"%(version_tag,index)
    print("[started][job] %s, # seeds: %d"%(version,iposts_records.shape[0]))
    iposts_records=iposts_records.to_dict(orient='records')
    
    simX=SimX(platform,domain,scenario,model_identifier)
    simX.set_metadata()
    simX.set_user_metadata()
    
    
    simX._run_simulate(iposts_records,version)
    print("[completed][job] %s"%version)


start = time.time()
"""
Load the simulation parameters
"""
parser = argparse.ArgumentParser(description='Simulation Parameters')
parser.add_argument('--config', dest='config_file_path', type=argparse.FileType('r'))
args = parser.parse_args()

config_json=json.load(args.config_file_path)
print(config_json)
platform = config_json["PLATFORM"]
domain = config_json["DOMAIN"]
scenario = int(config_json["SCENARIO"])
##no_trials = int(config_json["NO_TRIALS"])
version_tag = int(config_json["VERSION_TAG"])
input_posts_file_location=config_json["INPUT_SEEDS_FILE_PATH"]
model_identifier=config_json["MODEL_IDENTIFIER"]
#########



print("SOCIALSIM DECEMBER CHALLENGE: SIMULATION")
print("VERSION: %s" % version_tag)
print("------------ platform: %s. domain: %s. version: %s"%(platform,domain,version_tag))





### reset simulation output dirs.
output_dir="./output/%s/%s/%s/"%(platform,domain,model_identifier)
reset_dir(output_dir)
print("[reset] output dir: %s"%output_dir)

       

## loading input seeds
print("[input posts] Reading..")
# iposts=_get_input_posts(input_posts_file_location)
# print("[input posts] Done, # posts: %d"%iposts.shape[0])


iposts_array=_get_input_messages(input_posts_file_location)
##NUMBER_OF_SEED_CHUNKS=len(iposts_array)
Parallel(n_jobs=NUMBER_OF_SEED_CHUNKS)(delayed(_run)([iposts_array[index],index]) for index in range(NUMBER_OF_SEED_CHUNKS))


end = time.time()
elapsed=float(end - start)/60
print("Elapsed %0.2f minutes."%elapsed)

    

