## source activate pnnl_socialsim
## python run_cascade.py --config ./metadata/configs/twitter_cve_S2.json
import numpy as np
import pandas as pd
import sys,os
# from random import choices
import random
from datetime import datetime as dt
from datetime import datetime, timedelta, date
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
import pickle


from libs.lib_conversation import *
from joblib import Parallel, delayed

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
        
def daterange(start_date, end_date, delta=timedelta(days=7)):
    currentDate = start_date
    while currentDate < end_date:
        yield currentDate
        currentDate += delta
        
def del_dir(x_dir):
    if(os.path.exists(x_dir)==True):
        try:
            shutil.rmtree(x_dir)
            print("Deleted. %s"%x_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
            
def create_dir(x_dir):
    if not os.path.exists(x_dir):
        os.makedirs(x_dir)
        print("Created new dir. %s"%x_dir)
        


def reset_dir(x_dir):
    del_dir(x_dir)
    create_dir(x_dir)


def _get_input_messages(sim_data_shares,sim_data_ousers,sim_data_nusers):
        
    input_messages_dict={}
    for infoID in info_ids:#['covid']:#
        sim_data_shares_infoID=sim_data_shares[infoID]
        sim_data_ousers_infoID=sim_data_ousers[infoID]
        sim_data_nusers_infoID=sim_data_nusers[infoID]
        
        assert(len(sim_data_shares_infoID)==len(sim_data_ousers_infoID))
        assert(len(sim_data_shares_infoID)==len(sim_data_nusers_infoID))
        
        for step in range(len(sim_data_shares_infoID)):
            infoID_step=infoID+"_step_%d"%step
            iposts_time=sim_days[step]
            sim_data_tuple=(sim_data_shares_infoID[step],sim_data_ousers_infoID[step],sim_data_nusers_infoID[step],iposts_time)
            input_messages_dict.setdefault(infoID_step,sim_data_tuple)
    return input_messages_dict



### simulation counts loading
def _load(ktag,model_identifier):

    blend_output_path_shares = "/data/CP5_MCAS/MCAS/ml_output/{0}_{1}/{2}/local_topics/{3}/{4}".format(domain, scenario, platform, ktag, model_identifier)
    print("\n[Local Topics] %s loading.."%ktag)
    with open(blend_output_path_shares+'/simulations_data.pkl.gz', 'rb') as fd:
        sim_data_shares=pickle.load(fd)

    blend_output_path_shares = "/data/CP5_MCAS/MCAS/ml_output/{0}_{1}/{2}/global_topics/{3}/{4}".format(domain, scenario, platform, ktag, model_identifier)
    if os.path.exists(blend_output_path_shares):
        print("[Global Topics] %s loading.."%ktag)
        with open(blend_output_path_shares+'/simulations_data.pkl.gz', 'rb') as fd:
            gsim_data_shares=pickle.load(fd)
        sim_data_shares.update(gsim_data_shares)
    print(sim_data_shares)
    return sim_data_shares
            


def _run(args):
    iposts_infoid=args[0].split("_step_")[0]
    iposts_infoid_block=int(args[0].split("_step_")[1])
    iposts_infoid_label=iposts_infoid.replace("/","_")
    iposts_records=args[1]

    version="%s-step-%s-%s"%(iposts_infoid_label,version_tag,iposts_infoid_block)
    print("[started][job] %s, # seeds: %d, # old users: %d, # new users: %d"%(version,iposts_records[0],iposts_records[1],iposts_records[2],))
    ##iposts_records=iposts_records.to_dict(orient='records')
    
    
    simX=SimX(platform,domain,scenario,model_identifier,iposts_infoid)
    simX.set_metadata()
    simX.set_user_metadata()
    
    for i in range(1,sim_num_trials+1):
        simX._run_simulate(iposts_records,"V%d-%s"%(i,version))
        print("[completed][job] %s"%version)

def _run_model(model_identifier):
    print("\n\n%s\n\n"%model_identifier)
    ### reset simulation output dirs.
    output_dir="./output/%s/%s/%s/%s/"%(platform,domain,scenario,model_identifier)
    reset_dir(output_dir)
    print("[reset] output dir: %s"%output_dir)



    ## loading input seeds
    print("[input posts] Reading..")
    sim_data_shares=_load('total_shares',model_identifier)
    sim_data_ousers=_load('old_users',model_identifier)
    sim_data_nusers=_load('new_users',model_identifier)


    input_messages_dict=_get_input_messages(sim_data_shares,sim_data_ousers,sim_data_nusers)

    block_sum=0;block_count=0;
    for k,v in input_messages_dict.items():
        print(k,v)
        block_sum+=v[0]
        block_count+=1
    print("# shares: ",block_sum,block_count)


    Parallel(n_jobs=block_count)(delayed(_run)([k,v]) for k,v in input_messages_dict.items())
    # for k,v in input_messages_dict.items():
    #     _run([k,v])

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
model_identifiers = config_json["MODEL_IDENTIFIER"]
sim_num_trials = config_json["NUM_SIM_TRIALS"]
start_sim_period = config_json['START_SIM_PERIOD']
end_sim_period = config_json['END_SIM_PERIOD']
sim_start_date=datetime.strptime(start_sim_period,"%Y-%m-%d")
sim_end_date=datetime.strptime(end_sim_period,"%Y-%m-%d")
oneD=timedelta(days=1)
kgap=1
if scenario.startswith("weekly_"):
    kgap=7

sim_days=[]
for sim_day in daterange(sim_start_date, sim_end_date+oneD,delta=timedelta(days=kgap)):
    sim_day_text=sim_day.strftime("%Y-%m-%d")
    sim_days.append(sim_day_text)
oneD=timedelta(days=1)



version_tag = config_json["VERSION_TAG"]
##input_posts_file_location=config_json["INPUT_SEEDS_FILE_PATH"]
info_ids_path = config_json['INFORMATION_IDS']
info_ids_path = info_ids_path.format(platform)

### Load information IDs
info_ids = pd.read_csv(info_ids_path, header=None)
info_ids.columns = ['informationID']
info_ids = sorted(list(info_ids['informationID']))
print(len(info_ids),info_ids)
#########








print("DARPA SOCIALSIM SIMULATION")
print("------------ platform: %s. domain: %s. scenario: %s. version: %s"%(platform,domain,scenario,version_tag))



for model_identifier in model_identifiers:
    _run_model(model_identifier)
    

end = time.time()
elapsed=float(end - start)/60
print("Elapsed %0.2f minutes."%elapsed)

    

