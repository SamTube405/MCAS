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
prediction_type = config_json['PREDICTION_TYPE']
start_sim_period = config_json['start_sim_period']
end_sim_period = config_json['end_sim_period']
print("Blending ----- ",domain,platform,prediction_type)

info_ids_path = config_json['INFORMATION_IDS']
info_ids_path = info_ids_path.format(domain)

### Load information IDs
info_ids = pd.read_csv(info_ids_path, header=None)
info_ids.columns = ['informationID']
info_ids = sorted(list(info_ids['informationID']))
info_ids = ['informationID_'+x if 'informationID' not in x else x for x in info_ids]
print(len(info_ids),info_ids)

output_path = "./ml_output/{0}/{1}/{2}/".format(domain, platform, prediction_type)
blend_output_path = "./ml_output/{0}/{1}/{2}/BLEND_{3}_{4}/".format(domain, platform, prediction_type,start_sim_period,end_sim_period)
reset_dir(blend_output_path)

### Gperformance plots.
performance_data=[]
sim_data={}
for xmodel_name in glob("%s*/"%output_path):
    xmodel_name_array=xmodel_name.split("/")[-2].split("_")
    #print(xmodel_names_array)
    if len(xmodel_name_array)>3:
        if (xmodel_name_array[1]==start_sim_period) & (xmodel_name_array[2]==end_sim_period):
            print("Into the blender: ",xmodel_name)
            Gperformance=pd.read_pickle(xmodel_name+'Gperformance.pkl.gz')
            Gperformance['FILE_NAME']=xmodel_name
            
            with open(xmodel_name+'simulations_data.pkl.gz', 'rb') as fd:
                sim_data_=pickle.load(fd)
                sim_data.setdefault(xmodel_name,sim_data_)
            #print(Gperformance.head())
            performance_data.append(Gperformance)
performance_data=pd.concat(performance_data)

idx = performance_data.groupby(['informationID'])['RMSE'].transform(min) == performance_data['RMSE']
performance_data_blended=performance_data[idx]
print("\n\n Blended performance output ---------------")
print(performance_data_blended)
performance_data_blended.to_pickle("%sGperformance.pkl.gz"%blend_output_path)

sim_data_blended={}
def getBlendedSimulation(row):
    sim_array=sim_data[row['FILE_NAME']][row['informationID']]
    print(row['FILE_NAME'],row['informationID'],sim_array)
    sim_data_blended.setdefault(row['informationID'],sim_array)
print("\n\n Blended simulation output ---------------")    
T=performance_data_blended.apply(getBlendedSimulation,axis=1)
### Save blended simulation results
pickle.dump(sim_data_blended, open(blend_output_path+'simulations_data.pkl.gz', 'wb'))
    



