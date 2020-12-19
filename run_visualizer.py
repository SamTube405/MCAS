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
output_path_plots = "{0}plots".format(output_path_)
reset_dir(output_path_plots)


### Timeseries plots.
with open(output_path_+'gt_data.pkl.gz', 'rb') as fd:
    gt_data=pickle.load(fd)
    
with open(output_path_+'simulations_data.pkl.gz', 'rb') as fd:
    sim_data=pickle.load(fd)
    
with open(output_path_+'replay_simulations_data.pkl.gz', 'rb') as fd:
    replay_sim_data=pickle.load(fd)
    
with open(output_path_+'sampling_simulations_data.pkl.gz', 'rb') as fd:
    sampling_sim_data=pickle.load(fd)
    
for info_id in info_ids:
    print("Timeseries plot, %s"%info_id)
    fig, ax = plt.subplots(figsize=(10,6))

    plt.plot(gt_data[info_id],label='GT',color='black',lw=4)
    plt.plot(sim_data[info_id],label='MCAS',color='red',linestyle='-',lw=4)
    plt.plot(replay_sim_data[info_id],label='Replay',color='blue',linestyle='--',lw=4)
    plt.plot(sampling_sim_data[info_id],label='Sampling',color='cyan',linestyle=':',lw=4)

    plt.legend()
    plt.xlabel("Time (Days)",fontSize=15)
    plt.ylabel("# %ss"%prediction_type,fontSize=15)
    plt.yscale('log',basey=10)
    info_id_=info_id.replace('informationID_','')
    plt.title("Topic: %s"%info_id_,fontSize=14,fontweight='bold')
    
    times = [(start_sim_period+timedelta(days=i)).strftime("%m-%d") for i in range(sim_days)]

    ax.set_xticks(np.arange(sim_days+1))
    ax.set_xticklabels(times)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig('{0}/{1}_ts.pdf'.format(output_path_plots,info_id.replace("/","_")))
    
    
### Gperformance plots.
Gperformance=pd.read_pickle(output_path_+'Gperformance.pkl.gz')
BRperformance=pd.read_pickle(output_path_+'BRperformance.pkl.gz')
BSperformance=pd.read_pickle(output_path_+'BSperformance.pkl.gz')
performance_data=pd.concat([Gperformance,BRperformance,BSperformance])

hue_order_=[model_id,'Replay','Sampling']
hue_cs_=['red','blue','cyan']
hue_cs_=sns.color_palette(hue_cs_)
info_ids_=[x.replace('informationID_','') if 'informationID_' in x else x for x in info_ids]
for mea in ['APE','RMSE','NRMSE','SMAPE']:
    print("Performance plot, %s"%mea)
    g=sns.catplot(x='informationID',y=mea,hue='MODEL',
                  order=info_ids,
                hue_order=hue_order_,
                  palette=hue_cs_,
                kind='bar',height=6,aspect=2,legend=False,data=performance_data)
    g.set_xticklabels(info_ids_)
    plt.xticks(rotation=90)
    plt.xlabel("Topic",fontSize=15)
    plt.ylabel(mea,fontSize=15)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('{0}/{1}_performance.pdf'.format(output_path_plots,mea))
    
print('\nSimulation files for {0} stored at'.format(model_id), output_path_plots)


