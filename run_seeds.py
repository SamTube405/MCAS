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
    
def get_inode(props):
    degV=props.iloc[0]['degreeV']
    iNodes0=sum(degV)
    iNodes=iNodes0
    for l in props.index:
        if l>0:
            degV=props.iloc[l]['degreeV']
            iNodes+=(sum(degV))
        #print(l)
    iNodes0=iNodes0/iNodes
    #print(iNodes0)
    return iNodes0

# def get_virality(seed_count,response_count):
#     if platform=="twitter":
#         isViral=False
#         if (seed_count>0) & (response_count>0):
#             isViral=((response_count/seed_count)>18)
#         elif (seed_count==0) & (response_count>0):
#             isViral=True
#     elif platform=="youtube":
#         isViral=True
#     return isViral

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
scenario = config_json["SCENARIO"]
model_identifier = config_json["MODEL_IDENTIFIER"]
start_sim_period = config_json['START_SIM_PERIOD']
end_sim_period = config_json['END_SIM_PERIOD']
sim_start_date=datetime.strptime(start_sim_period,"%Y-%m-%d")
sim_end_date=datetime.strptime(end_sim_period,"%Y-%m-%d")
oneD=timedelta(days=1)

info_ids_path = config_json['INFORMATION_IDS']
##info_ids_path = info_ids_path.format(domain)

### Load information IDs
info_ids = pd.read_csv(info_ids_path, header=None)
info_ids.columns = ['informationID']
info_ids = sorted(list(info_ids['informationID']))
info_ids = ['informationID_'+x if 'informationID' not in x else x for x in info_ids]
print(len(info_ids),info_ids)

# ### GT Test
# blend_output_path_seed = "/data/CP5_MCAS/MCAS/ml_output/{0}_{1}/{2}/seed/BLEND_{3}_{4}".format(domain, scenario, platform, start_sim_period,end_sim_period)
# blend_output_path_response = "/data/CP5_MCAS/MCAS/ml_output/{0}_{1}/{2}/response/BLEND_{3}_{4}".format(domain, scenario, platform, start_sim_period,end_sim_period)

# print("[Seed counts] loading..")
# with open(blend_output_path_seed+'/gt_data_simulations.pkl.gz', 'rb') as fd:#simulations_data.pkl.gz
#     sim_data_seed=pickle.load(fd)

# print("[Response counts] loading..")
# with open(blend_output_path_response+'/gt_data_simulations.pkl.gz', 'rb') as fd:#simulations_data.pkl.gz
#     sim_data_response=pickle.load(fd)
    

blend_output_path_seed = "/data/CP5_MCAS/MCAS/ml_output/{0}_{1}/{2}/seed/{3}_{4}_{5}".format(domain, scenario, platform, model_identifier,start_sim_period,end_sim_period)
blend_output_path_response = "/data/CP5_MCAS/MCAS/ml_output/{0}_{1}/{2}/response/{3}_{4}_{5}".format(domain, scenario, platform,model_identifier,start_sim_period,end_sim_period)

print("[Seed counts] loading..")
with open(blend_output_path_seed+'/simulations_data.pkl.gz', 'rb') as fd:#gt_data_simulations.pkl.gz
    sim_data_seed=pickle.load(fd)

print("[Response counts] loading..")
with open(blend_output_path_response+'/simulations_data.pkl.gz', 'rb') as fd:#gt_data_simulations.pkl.gz
    sim_data_response=pickle.load(fd)



# print(sim_data_seed)
global_event_count=0
global_seed_count=0
global_ml_count=0
for info_id in info_ids:
    infoID_label=info_id.replace("/","_")
    blend_output_path_seed_cascade = "./metadata/probs/{0}/{1}/{2}/{3}".format(platform, domain,scenario,infoID_label)
    reset_dir(blend_output_path_seed_cascade)
    filePath="%s/input_seeds_responses_%s.csv"%(blend_output_path_seed_cascade,model_identifier) 
    if os.path.exists(filePath):
        os.remove(filePath)
        print("Existing file deleted.")
    fd=open(filePath,'a')
    
    cascade_props_prob_degree=pd.read_pickle("./metadata/probs/%s/%s/%s/%s/cascade_props_prob_degree.pkl.gz"%(platform,domain,scenario,infoID_label))
    degreeList=np.array(cascade_props_prob_degree.loc[0]['udegreeV'])
    degreeProbList=np.array(cascade_props_prob_degree.loc[0]["probV"])
    iNode0=get_inode(cascade_props_prob_degree)

    index=0
    seed_data=sim_data_seed[info_id]
    response_data=sim_data_response[info_id]
    
#     total_seed_count=sum(seed_data)
#     total_response_count=sum(response_data)
#     if total_seed_count>0:
#         seed_avg_responses=int(total_response_count/total_seed_count)
#         print(seed_avg_responses)
    
    local_seed_count=0
    local_response_count=0
    local_event_count=0
    local_ml_count=0
    for sim_day in daterange(sim_start_date, sim_end_date+oneD):
        sim_day_text=sim_day.strftime("%Y-%m-%d")
        seed_count=int(seed_data[index])
        response_count=int(response_data[index])
       
        
        ##print("Day: %s, InfoID: %s, # seeds: %d, # responses: %d"%(sim_day_text,info_id, seed_count, response_count))
        #cascade_child_count=seed_count+response_count
        local_ml_count+=seed_count+response_count
        index+=1
        

#         isViral=get_virality(seed_count,response_count)

            
#         if isViral:
#             k=min(2,len(degreeList)-1)
#             degreeList_=degreeList[k:]
#             degreeProbList_=degreeProbList[k:]
#             degreeProbList_=degreeProbList_/np.sum(degreeProbList_)
#             ##print(degreeList_,degreeProbList_)
#         else:
#             degreeList_=degreeList
#             degreeProbList_=degreeProbList
            
        print("Day: %s, InfoID: %s, # messages: %d (%0.2f)"%(sim_day_text,info_id, seed_count+response_count,iNode0))
            
        response_count=int(response_count*iNode0)
        
        cascade_child_count_dict={}
        for j in range(10):
            cascade_child_count=seed_count+response_count
            cascade_child_count_array=[]
            while (cascade_child_count>0): 
                sampled_degree = np.random.choice(a=degreeList, p=degreeProbList)
                cascade_child_count-=(sampled_degree+1)
                cascade_child_count_array.append(sampled_degree)
            cascade_child_count_dict.setdefault(cascade_child_count,cascade_child_count_array)
            
            
        cascade_child_count_max=max(list(cascade_child_count_dict.keys()))
        cascade_child_count_array=cascade_child_count_dict[cascade_child_count_max]
        
        for sampled_degree in cascade_child_count_array:
            seed_identifier = "seed_%16x"%random.getrandbits(64)
            seed_user_id=None
            local_seed_count+=1
            local_event_count+=1

            local_response_count+=sampled_degree 
            local_event_count+=sampled_degree
                    
            fd.write("%s,%s,%s,%d,%s\n"%(sim_day_text,seed_identifier,seed_user_id,sampled_degree,info_id))  
            
    global_event_count+=local_event_count
    global_ml_count+=local_ml_count
    global_seed_count+=local_seed_count
        
        
    print("Expected: ",int(local_event_count/iNode0)," ML predicted: ",local_ml_count)#cascade_seeds_count,
    print("# Seeds: ",local_seed_count)
    print("# responses ",local_response_count," required: ",local_event_count-local_seed_count)
        
        
    print("Saved seeds at %s"%filePath)

print("Total # Seeds: ",global_seed_count)
print("Total # Events (Reduced by Level 0 child nodes): ",global_event_count)
print("Total # ML Events: ",global_ml_count)

   



