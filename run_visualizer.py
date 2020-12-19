import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date
import argparse
import json
import shutil,os
from os import listdir
from os.path import isfile, join
import pickle

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

start_sim_period = config_json['start_sim_period']
end_sim_period = config_json['end_sim_period']
start_sim_period=datetime.strptime(start_sim_period,"%Y-%m-%d")
end_sim_period=datetime.strptime(end_sim_period,"%Y-%m-%d")
sim_days = end_sim_period - start_sim_period
sim_days = sim_days.days + 1


n_in = config_json['time_window']['n_in']
n_out = config_json['time_window']['n_out']

info_ids_path = config_json['INFORMATION_IDS']
info_ids_path = info_ids_path.format(domain)

model_id = config_json['MODEL_PARAMS']

### Load information IDs
info_ids = pd.read_csv(info_ids_path, header=None)
info_ids.columns = ['informationID']
info_ids = sorted(list(info_ids['informationID']))
info_ids = ['informationID_'+x if 'informationID' not in x else x for x in info_ids]
print(len(info_ids),info_ids)

output_path = "./ml_output/{0}/{1}/{2}/".format(domain, platform, prediction_type)
output_dir = "{0}_{1}_{2}_{3}-to-{4}_{5}".format(model_id, str(start_sim_period.strftime("%Y-%m-%d")), str(end_sim_period.strftime("%Y-%m-%d")), str(n_in), str(n_out), version)
output_path_ = output_path+output_dir+'/'
output_path_plots = "{0}/plots".format(output_path_)
reset_dir(output_path_plots)


Gperformance=pd.read_pickle(output_path_+'Gperformance.pkl.gz')

with open(output_path_+'gt_data.pkl.gz', 'rb') as fd:
    gt_data=pickle.load(fd)
    
with open(output_path_+'simulations_data.pkl.gz', 'rb') as fd:
    sim_data=pickle.load(fd)
    
for info_id in info_ids:
    
    fig, ax = plt.subplots(figsize=(10,6))

    plt.plot(gt_data[info_id],label='GT',color='black',lw=4)
    plt.plot(sim_data[info_id],label='MCAS',color='orange',linestyle='-',lw=4)

    plt.legend()
    plt.xlabel("Time (Days)",fontSize=15)
    plt.ylabel("# Tweets",fontSize=15)
    plt.yscale('log',basey=10)
    plt.title("Narrative: %s"%info_id,fontSize=14,fontweight='bold')
    
    times = [(start_sim_period+timedelta(days=i)).strftime("%m-%d") for i in range(sim_days)]

    ax.set_xticks(np.arange(sim_days+1))
    ax.set_xticklabels(times)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig('{0}/{1}.pdf'.format(output_path_plots,info_id.replace("/","_")))



