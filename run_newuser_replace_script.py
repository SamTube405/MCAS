#!/usr/bin/env python
# coding: utf-8

### V3: difference is only new users are introduced to cascades.

# python run_newuser_replace_script.py

import pandas as pd
import numpy as np
import random
import networkx as nx
import math
from joblib import Parallel, delayed
import time, math
import json
import glob
import os
import pickle
from datetime import datetime, timedelta, date
from collections import Counter

from libs.helper_funcs import *

""" Parameters to Change in Script """
columns=['rootID', 'informationID', 'actionType', 'nodeID', 'nodeTime','nodeUserID','parentID', 'rootUserID', 'parentUserID','platform']

platform = "twitter"
seedType = "tweet"
responseType = "retweet"

# platform = "youtube"
# seedType = "video"
# responseType = "comment"

# platform = "jamii"
# seedType = "post"
# responseType = "comment"

# platform = "reddit"
# seedType = "post"
# responseType = "comment"

### Simulation periods
start_sim = "2020-11-09"
start_sim = "2020-11-30"

end_sim = datetime.strptime(end_sim,"%Y-%m-%d") + timedelta(days=1)
end_sim = end_sim.strftime("%Y-%m-%d")

### period for previous users file
t3= datetime.strptime(start_sim,"%Y-%m-%d") - timedelta(days=1)
t3 = t3.strftime("%Y-%m-%d")

## Model parameters and directories
# model_id = 'MCAS-BLENDED-INTERNAL'
# model_id_ = "MCAS-BLENDED-INTERNAL"

# model_id = "MCAS-BLENDED-EXOG"
# model_id_ = "MCAS-BLENDED-EXOG"

# model_id = "MCAS-ENSEMBLE"
# model_id_ = "MCAS-ENSEMBLE"

model_id = "MCAS-META-ENSEMBLE"
model_id_ = "MCAS-META-ENSEMBLE"

sim_file_domain = "domain_name"

sim_file_name = "{0}_v1.pkl.gz".format(model_id.lower())

final_sim_file_name = sim_file_name

### Path to informationIDs
narratives_path = "./metadata/configs/infoids_{0}.txt".format(platform)

### Path to load cascade output file
sim_file_path = "./newuser_module/Simulation_cascade_output/{0}/{1}/{2}/{3}".format(sim_file_domain, platform, model_id, sim_file_name)
### Path to save simulation file after new/old user replacements
simulation_output_path = "./newuser_module/Simulations_new_user_output/{0}/{1}/{2}/".format(sim_file_domain,platform, model_id)
simulation_output_conflict_path = "./newuser_module/Simulations_new_user_output/{0}/{1}/{2}/conflicts_{3}/".format(sim_file_domain,platform, model_id, final_sim_file_name.replace(".pkl.gz",""))
### Create directory
create_dir(simulation_output_path)
simulation_output_path = simulation_output_path+final_sim_file_name.replace(".pkl.gz","")

### Path to store intermidiate narrative results (all files to be removed after script completion)
simulation_per_narrative_path = './newuser_module/Simulations_extradata/'
"""End of Parameters Change"""

model_id = model_id_
""" Load new/old user predictions """
if (platform == "twitter" or platform == "youtube"): 
    newuser_sim_file1 = "./ml_output/{0}/{1}/local_topics/new_users/{2}/simulations_data.pkl.gz".format(sim_file_domain, platform, model_id)
    newuser_sim_file2 = "./ml_output/{0}/{1}/global_topics/new_users/{2}/simulations_data.pkl.gz".format(sim_file_domain, platform, model_id)
    newuser_sim_file = [newuser_sim_file1, newuser_sim_file2]
else:
    newuser_sim_file = ["./ml_output/{0}/{1}/local_topics/new_users/{2}/simulations_data.pkl.gz".format(sim_file_domain, platform, model_id)]
    
if (platform == "twitter" or platform == "youtube"): 
    olduser_sim_file1 = "./ml_output/{0}/{1}/local_topics/old_users/{2}/simulations_data.pkl.gz".format(sim_file_domain, platform, model_id)
    olduser_sim_file2 = "./ml_output/{0}/{1}/global_topics/old_users/{2}/simulations_data.pkl.gz".format(sim_file_domain, platform, model_id)
    olduser_sim_file = [olduser_sim_file1, olduser_sim_file2]
else:
    olduser_sim_file = ["./ml_output/{0}/{1}/local_topics/old_users/{2}/simulations_data.pkl.gz".format(sim_file_domain, platform, model_id)]
    
print()
print(newuser_sim_file)
print(olduser_sim_file)

simulation_newusers_paths = newuser_sim_file
simulation_oldusers_paths = olduser_sim_file

### Check granularity
if "daily" in sim_file_domain:
    freq = "D"
else:
    freq="W-MON"
    
### Load new/old user predictions
results=[]
for simulation_newusers_path in simulation_newusers_paths:
    with open(simulation_newusers_path, 'rb') as fd:
        act_users=pickle.load(fd)
    time_idx = pd.date_range(start_sim, end_sim, freq=freq)
    time_idx=time_idx[:-1]

    for k, v in act_users.items():
        clean_k = k.replace("informationID_", "")
        tmp=pd.DataFrame(time_idx, columns=['nodeTime'])
        tmp['informationID'] = clean_k
        tmp['new_users'] = v
        tmp['platform'] = platform
        results.append(tmp)    
df_nuser = pd.concat(results, ignore_index=True)
df_nuser['nodeTime'] = df_nuser['nodeTime'].astype(str)

results=[]
for simulation_oldusers_path in simulation_oldusers_paths:
    with open(simulation_oldusers_path, 'rb') as fd:
        act_users=pickle.load(fd)
    time_idx = pd.date_range(start_sim, end_sim, freq=freq)
    time_idx=time_idx[:-1]

    for k, v in act_users.items():
        clean_k = k.replace("informationID_", "")
        tmp=pd.DataFrame(time_idx, columns=['nodeTime'])
        tmp['informationID'] = clean_k
        tmp['old_users'] = v
        tmp['platform'] = platform
        results.append(tmp)    
df_ouser = pd.concat(results, ignore_index=True)
df_ouser['nodeTime'] = df_ouser['nodeTime'].astype(str)
""" Finish Loading New/Old user predictions"""

"""Start Script"""
### Remove any files within the temporary directory
for f in os.listdir(simulation_per_narrative_path):
    os.remove(os.path.join(simulation_per_narrative_path, f))
    
### Load frames list
infoids = pd.read_csv(narratives_path, header=None)
infoids.columns = ['informationID']
infoids = sorted(list(infoids['informationID']))

### Load previous users (informationID -> nodeUseID)
# prev_users = pd.read_pickle(previous_users_path)

### Load simulation file
df_sim_orig = pd.read_pickle(sim_file_path)
df_sim_orig=convert_datetime(df_sim_orig, True)
df_sim_orig['informationID'] = df_sim_orig['informationID'].apply(lambda x: x.replace("informationID_", ""))
df_sim_orig = df_sim_orig.loc[df_sim_orig['informationID'].isin(infoids)].reset_index(drop=True)
df_sim_orig = df_sim_orig.query("nodeTime >= @start_sim and nodeTime < @end_sim").reset_index(drop=True)
df_sim_orig['nodeTime'] = pd.to_datetime(df_sim_orig['nodeTime'])
df_sim_orig["nodeTime"] = df_sim_orig['nodeTime'].dt.strftime('%Y-%m-%d')
df_sim_orig["nodeTime"] = df_sim_orig['nodeTime'].astype(str)
df_sim_orig = df_sim_orig.loc[df_sim_orig['platform'] == platform].reset_index(drop=True)
if "version" in df_sim_orig:
    df_sim_orig = df_sim_orig.drop(columns=["version"])
df_sim = df_sim_orig.copy()
print('Processing:', sim_file_path)
infoids = sorted(df_sim["informationID"].unique())

""" Compute out degrees """
deg_dict = getOutDegrees(df_sim)
print("Computing nodeUserIDs in degrees...")

### Start New User Replacement
print('Start '+platform+' New User Replacement...')
start_time = time.time()
Parallel(n_jobs=4)(delayed(newuser_replacement_v3)(df_sim, df_nuser, deg_dict,infoid, platform=platform,seedType=seedType,responseType=responseType, tmp_path=simulation_per_narrative_path, conflict_path="") for infoid in infoids)
print("%s new user replacement finished in--- %s seconds ---" % (platform,time.time() - start_time))

### Load new user outputs per narrative
files_per_narrative = glob.glob(simulation_per_narrative_path+'*.pkl.gz')
print('Files stored for new user replacement:',len(files_per_narrative))
concat = []
for file in files_per_narrative:
    tmp = pd.read_pickle(file)
    concat.append(tmp)  
newuser_df = pd.concat(concat, ignore_index=True)
newuser_df=newuser_df.sort_values('nodeTime').reset_index(drop=True)

### Remove temporary new user files
for file in files_per_narrative:
    os.remove(file)

### Save final simulation output
final_df = newuser_df.copy()
final_df.to_pickle(simulation_output_path+'_final_output.pkl.gz')

print(platform, model_id)
print('Stats after replacement:')
print('# Events:', len(final_df.loc[final_df['platform']==platform]))
print('# Seeds:', len(final_df.loc[final_df['actionType'] == seedType]), '# Seed Users:',final_df.loc[final_df['actionType'] == seedType]['nodeUserID'].nunique())
print('# Responses:', len(final_df.loc[final_df['actionType'] != seedType]), '# Response Users:', final_df.loc[final_df['actionType'] != seedType]['nodeUserID'].nunique())
print()
