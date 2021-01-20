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

##NUMBER_OF_SEED_CHUNKS=100

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
    
    
#     infoIDs=input_messages['informationID'].unique()
#     print("Presence: ",infoIDs.shape[0],infoIDs)
    
    input_messages_dict={}
    for infoID in info_ids:
        infoID_label=infoID.replace("/","_")
        
        
        ipath=path.format(platform,domain,scenario,infoID_label)
        try:
            input_messages=pd.read_csv(ipath,header=None)
            input_messages.columns=['nodeTime','nodeID','nodeUserID','iDegree','informationID']
        except pd.errors.EmptyDataError:
            continue
        
    
        if input_messages.shape[0]>0:
            print(infoID,input_messages.shape[0])
            input_messages_=input_messages.query('informationID==@infoID')
            input_messages_.reset_index(inplace=True)
            prevj=0;
            nextj=0
            input_messages_array=[]
            while nextj<input_messages_.shape[0]:
                prevj=nextj
                nextj+=1000
                nextj=min(nextj,input_messages_.shape[0])
                input_messages_array.append(input_messages_.iloc[prevj:nextj])

            ##input_messages_array=np.array_split(input_messages_, 10)
            ##np.split(input_messages_, np.arange(1000,len(a),size))
            for i in range(len(input_messages_array)):
                im=input_messages_array[i]
                infoID_im=infoID+"_block_%d"%i
                input_messages_dict.setdefault(infoID_im,im)
    ###input_messages=pd.concat(input_messages_array)
    return input_messages_dict


def _run(args):
    iposts_infoid=args[0].split("_block_")[0]
    iposts_infoid_block=args[0].split("_block_")[1]
    iposts_infoid_label=iposts_infoid.replace("/","_")
    iposts_records=args[1]

    version="%s-block-%s-%s"%(iposts_infoid_label,version_tag,iposts_infoid_block)
    print("[started][job] %s, # seeds: %d"%(version,iposts_records.shape[0]))
    iposts_records=iposts_records.to_dict(orient='records')
    
    simX=SimX(platform,domain,scenario,iposts_infoid)
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
scenario = config_json["SCENARIO"]
version_tag = config_json["VERSION_TAG"]
input_posts_file_location=config_json["INPUT_SEEDS_FILE_PATH"]
info_ids_path = config_json['INFORMATION_IDS']
##info_ids_path = info_ids_path.format(domain)

### Load information IDs
info_ids = pd.read_csv(info_ids_path, header=None)
info_ids.columns = ['informationID']
info_ids = sorted(list(info_ids['informationID']))
info_ids = ['informationID_'+x if 'informationID' not in x else x for x in info_ids]
print(len(info_ids),info_ids)
#########



print("DARPA SOCIALSIM SIMULATION")
print("------------ platform: %s. domain: %s. scenario: %s. version: %s"%(platform,domain,scenario,version_tag))





### reset simulation output dirs.
output_dir="./output/%s/%s/%s/"%(platform,domain,scenario)
reset_dir(output_dir)
print("[reset] output dir: %s"%output_dir)

       

## loading input seeds
print("[input posts] Reading..")
input_messages_dict=_get_input_messages(input_posts_file_location)
block_sum=0;block_count=0;
for k,v in input_messages_dict.items():
    print(k,v.shape[0])
    block_sum+=v.shape[0]
    block_count+=1
print("# seeds block array: ",block_sum,block_count)

# # Parallel(n_jobs=NUMBER_OF_SEED_CHUNKS)(delayed(_run)([input_messages_array[index],index,False]) for index in range(NUMBER_OF_SEED_CHUNKS))
# NUMBER_OF_SEED_CHUNKS=len(info_ids)
Parallel(n_jobs=block_count)(delayed(_run)([k,v]) for k,v in input_messages_dict.items())
# #_run([input_messages_,0,isViralAtt])

end = time.time()
elapsed=float(end - start)/60
print("Elapsed %0.2f minutes."%elapsed)

    

