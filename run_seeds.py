import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import argparse
import json
import shutil,os
from os import listdir
from os.path import isfile, join
import pickle
from glob import glob
import random

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
        
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
#print(config_json)

# version = config_json['VERSION_TAG']
domain = config_json['DOMAIN']
platform = config_json['PLATFORM']
start_sim_period = config_json['START_SIM_PERIOD']
end_sim_period = config_json['END_SIM_PERIOD']
sim_start_date=datetime.strptime(start_sim_period,"%Y-%m-%d")
sim_end_date=datetime.strptime(end_sim_period,"%Y-%m-%d")
oneD=timedelta(days=1)

info_ids_path = config_json['INFORMATION_IDS']
info_ids_path = info_ids_path.format(domain)

### Load information IDs
info_ids = pd.read_csv(info_ids_path, header=None)
info_ids.columns = ['informationID']
info_ids = sorted(list(info_ids['informationID']))
info_ids = ['informationID_'+x if 'informationID' not in x else x for x in info_ids]
print(len(info_ids),info_ids)

#output_path = "./ml_output/{0}/{1}/{2}/".format(domain, platform, prediction_type)
blend_output_path_seed = "./ml_output/{0}/{1}/{2}/BLEND_{3}_{4}/".format(domain, platform, "seed",start_sim_period,end_sim_period)
blend_output_path_seed_inf = "./ml_output/{0}/{1}/{2}/BLEND_{3}_{4}/".format(domain, platform, "seed_inf",start_sim_period,end_sim_period)

print("[Seed counts] loading..")
with open(blend_output_path_seed+'simulations_data.pkl.gz', 'rb') as fd:
    sim_data_seed=pickle.load(fd)

print("[Seed Inf counts] loading..")
with open(blend_output_path_seed_inf+'simulations_data.pkl.gz', 'rb') as fd:
    sim_data_seed_inf=pickle.load(fd)

blend_output_path_seed_cascade = "./ml_output/{0}/{1}/{2}/BLEND_{3}_{4}/".format(domain, platform, "seed_cascade",start_sim_period,end_sim_period)
reset_dir(blend_output_path_seed_cascade)
filePath="%ssimulations_data.csv"%blend_output_path_seed_cascade  
if os.path.exists(filePath):
    os.remove(filePath)
    print("Existing file deleted.")
fd=open(filePath,'a')

# print(sim_data_seed)
global_event_count=0
for info_id in info_ids:
    

    ######
    index=0
    seed_viral=False
    seed_data=sim_data_seed[info_id]
    for sim_day in daterange(sim_start_date, sim_end_date+oneD):
        sim_day_text=sim_day.strftime("%Y-%m-%d")
        seed_count=int(seed_data[index])
        print("Day: %s, InfoID: %s, # seeds: %d"%(sim_day_text,info_id, seed_count))
        global_event_count+=seed_count
        index+=1
        
#         if seed_count<=0:
#             continue

        for seed_index in range(seed_count):
            seed_identifier = "seed_%16x"%random.getrandbits(64)
            seed_user_id=None
            fd.write("%s,%s,%s,%s,%s\n"%(sim_day_text,seed_identifier,seed_user_id,seed_viral,info_id))
            
            
    ######
    index=0
    seed_viral=True
    seed_data=sim_data_seed_inf[info_id]
    for sim_day in daterange(sim_start_date, sim_end_date+oneD):
        sim_day_text=sim_day.strftime("%Y-%m-%d")
        seed_count=int(seed_data[index])
        print("Day: %s, InfoID: %s, # seeds: %d"%(sim_day_text,info_id, seed_count))
        if seed_count<=0:
            continue
        global_event_count+=seed_count
        index+=1

        for seed_index in range(seed_count):
            seed_identifier = "seed_%16x"%random.getrandbits(64)
            seed_user_id=None
            fd.write("%s,%s,%s,%s,%s\n"%(sim_day_text,seed_identifier,seed_user_id,seed_viral,info_id))
            
            
print("Saved seeds at %s"%filePath)
print("Total # seeds: ",global_event_count)

   



