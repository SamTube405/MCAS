import os
import pandas as pd
import numpy as np
import argparse
import json
from treelib import Node, Tree, tree
from datetime import datetime, timedelta, date
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def datespan(startDate, endDate, delta=timedelta(days=7)):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta
        
from libs.lib_job_thread import *
from joblib import Parallel, delayed
import string 
import random
letters = list(string.ascii_lowercase)
def rand(stri):
    return random.choice(letters)

def create_dir(x_dir):
    if not os.path.exists(x_dir):
        os.makedirs(x_dir)
        print("Created new dir. %s"%x_dir)

class Node(object):
    def __init__(self, node_id,node_author,node_short_delay=0,node_long_delay=0):
        self.node_id=node_id
        self.node_author=node_author
        self.node_short_delay=node_short_delay
        self.node_long_delay=node_long_delay

    def get_node_id(self):
        return self.node_id
    
    def get_node_author(self):
        return self.node_author 
    
    def get_node_short_delay(self):
        return self.node_short_delay
    
    def get_node_long_delay(self):
        return self.node_long_delay
    
class Cascade(object):
    def __init__(self,platform,domain,scenario,infoID,cascade_records):
        self.platform=platform
        self.domain=domain
        self.scenario=scenario
        self.infoID=infoID
        infoID_label=infoID.replace("/","_")
        self.output_dir="./metadata/probs/%s/%s/%s/%s"%(self.platform,self.domain,self.scenario,infoID_label)
        create_dir(self.output_dir)
        
        self.pool = ThreadPool(128) 
        self.cascade_props=[]


        self.cascade_records=cascade_records
        self.cascade_records['actionType']='response'
        self.cascade_records.loc[self.cascade_records["nodeID"]==self.cascade_records["parentID"],"actionType"]="seed"
        print("# cascades: %d, # nodes: %d"%(self.cascade_records['rootID'].nunique(),self.cascade_records.shape[0]))

        
    def prepare_data(self):
        node_users=self.cascade_records[['nodeID','nodeTime']].drop_duplicates().dropna()
        
        node_users.columns=['parentID','parentTime']
        self.cascade_records=pd.merge(self.cascade_records,node_users,on='parentID',how='left')
        self.cascade_records.loc[self.cascade_records['parentID'].isna()==True,'parentID']=self.cascade_records['nodeID']
        self.cascade_records.loc[self.cascade_records['parentUserID'].isna()==True,'parentUserID']=self.cascade_records['nodeUserID']
        self.cascade_records.loc[self.cascade_records['parentTime'].isna()==True,'parentTime']=self.cascade_records['nodeTime']

        node_users.columns=['rootID','rootTime']
        self.cascade_records=pd.merge(self.cascade_records,node_users,on='rootID',how='left')
        self.cascade_records.loc[self.cascade_records['rootID'].isna()==True,'rootID']=self.cascade_records['parentID']
        
        self.cascade_records.loc[self.cascade_records['rootUserID'].isna()==True,'rootUserID']=self.cascade_records['parentUserID']
        self.cascade_records.loc[self.cascade_records['rootTime'].isna()==True,'rootTime']=self.cascade_records['parentTime']
        
        
        
        
        self.cascade_records["short_propagation_delay"]=self.cascade_records['nodeTime']-self.cascade_records['parentTime']
        self.cascade_records["long_propagation_delay"]=self.cascade_records['nodeTime']-self.cascade_records['rootTime']
        
        self.cascade_records.to_pickle("%s/cascade_records.pkl.gz"%(self.output_dir))
        
    def get_user_diffusion(self):
        responses=self.cascade_records.query('actionType=="response"')
        if responses.shape[0]<1:
            responses=self.cascade_records.query('actionType=="seed"')
        ##responses.loc[responses['isNew']==True,'nodeUserID']="new_"
        user_diffusion=responses.groupby(['parentUserID','nodeUserID']).size().reset_index(name='num_responses')
        user_diffusion_=responses.groupby(['parentUserID']).size().reset_index(name='total_num_responses')
        
        user_diffusion=pd.merge(user_diffusion,user_diffusion_,on='parentUserID',how='inner')
        user_diffusion['prob']=user_diffusion['num_responses']/user_diffusion['total_num_responses']
        user_diffusion.sort_values(['parentUserID','prob'],ascending=False,inplace=True)
        user_diffusion.to_pickle("%s/user_diffusion.pkl.gz"%(self.output_dir))

        
#         responses=self.cascade_records.query('actionType=="response"')
#         responses.loc[responses['isNew']==True,'nodeUserID']="new_"#+cascade_records_chunk['nodeUserID'].str.replace('[a-z]',rand)
#         user_diffusion0=responses.groupby(['parentUserID','nodeUserID']).size().reset_index(name='num_responses')

#         user_diffusion1=responses.groupby(['parentUserID']).size().reset_index(name='num_children')
#         user_diffusion1=pd.merge(user_diffusion0,user_diffusion1,on='parentUserID',how='inner')
#         user_diffusion1['prob_parent']=user_diffusion1['num_responses']/user_diffusion1['num_children']

#         user_diffusion2=responses.groupby(['nodeUserID']).size().reset_index(name='num_parents')
#         user_diffusion2=pd.merge(user_diffusion0,user_diffusion2,on='nodeUserID',how='inner')
#         user_diffusion2['prob_child']=user_diffusion2['num_responses']/user_diffusion2['num_parents']
#         user_diffusion2.drop(columns=['num_responses'],inplace=True)

#         user_diffusion=pd.merge(user_diffusion1,user_diffusion2,on=['parentUserID','nodeUserID'],how='inner')
#         user_diffusion['prob']=(user_diffusion['prob_parent']+user_diffusion['prob_child'])/2
#         user_diffusion.sort_values(['parentUserID','prob'],ascending=False,inplace=True)
        
#         user_diffusion.to_pickle("%s/user_diffusion.pkl.gz"%(self.output_dir))
    
        return user_diffusion
    
#     def get_decay_user_diffusion(self):
#         responses=self.cascade_records.query('actionType=="response"')
#         responses.loc[responses['isNew']==True,'nodeUserID']="new_"
        
#         pair_lifetime=responses.groupby(['parentUserID','nodeUserID'])['nodeTime'].min().reset_index(name='lifetime_min')
        
#         pair_lifetime=pd.merge(responses,pair_min_lifetime,on=['parentUserID','nodeUserID'],how='inner')
#         pair_lifetime['lifetime']=(pair_lifetime['nodeTime']-pair_lifetime['lifetime_min']).dt.days
#         pair_lifetime['lifetime_max']=(start_sim_period_date-pair_lifetime['lifetime_min']).dt.days
#         pair_lifetime=pair_lifetime[pair_lifetime['lifetime']>0]
        
#         pair_lifetime.groupby(['parentUserID','nodeUserID'])['lifetime'].apply(set)
            
        
    
    def get_user_spread_info(self):
        self.spread_info1=self.cascade_records.query('actionType=="seed"').groupby(['nodeUserID'])['nodeID'].nunique().reset_index(name="num_seeds")
        num_seed_users=self.spread_info1['nodeUserID'].nunique()
        print("# seed users: %d"%num_seed_users)

        self.spread_info2=self.cascade_records.query('actionType=="response"').groupby(['nodeUserID'])['nodeID'].nunique().reset_index(name="num_responses")
        print("# responding users: %d"%self.spread_info2['nodeUserID'].nunique())

        dataset_users=self.cascade_records[['nodeUserID','nodeID','actionType']].drop_duplicates()
        dataset_users_only_seed=dataset_users.query('actionType=="seed"')
        all_responses=self.cascade_records.query('actionType=="response"').groupby(['rootID'])['nodeID'].nunique().reset_index(name='num_responses')
        all_responses_with_users=pd.merge(all_responses,dataset_users_only_seed,left_on='rootID',right_on='nodeID',how='inner')
        dataset_responded_seeds=all_responses_with_users.groupby(['nodeUserID'])['rootID'].nunique().reset_index(name='num_responded_seeds')
        dataset_responded_vol=all_responses_with_users.groupby(['nodeUserID'])['num_responses'].sum().reset_index(name='num_responses_recvd')

        self.spread_info=pd.merge(self.spread_info1,self.spread_info2,on=['nodeUserID'],how='outer')
        self.spread_info.fillna(0,inplace=True)
        print("# Total users: %d"%self.spread_info['nodeUserID'].nunique())

        self.spread_info=pd.merge(self.spread_info,dataset_responded_seeds,on=['nodeUserID'],how='left')
        self.spread_info.fillna(0,inplace=True)

        self.spread_info=pd.merge(self.spread_info,dataset_responded_vol,on=['nodeUserID'],how='left')
        self.spread_info.fillna(0,inplace=True)

        self.spread_info['spread_score']=(self.spread_info['num_responded_seeds']/self.spread_info['num_seeds'])*self.spread_info['num_responses_recvd']
        self.spread_info.sort_values(by='spread_score',ascending=False,inplace=True)
        self.spread_info.set_index('nodeUserID',inplace=True)

                
        self.spread_info.to_pickle("%s/user_spread_info.pkl.gz"%(self.output_dir))
        return self.spread_info
       


    def get_cascade_tree(self,cascade_tuple):
        rootID=cascade_tuple[0]
        rootUserID=cascade_tuple[1]
        childNodes=cascade_tuple[2]
        
        cascadet=Tree()

        
        parent=Node(rootID,rootUserID,0,0)
        cascadet.create_node(rootID, rootID, data=parent)
        print(rootID,rootUserID,childNodes)
        for m in childNodes:
            comment_id=m[0]
            parent_post_id=m[1]
            child_author_id=m[2]
            short_delay=m[3]
            long_delay=m[4]
            child=Node(comment_id,child_author_id,short_delay,long_delay)

            try:
                parent_node=cascadet.get_node(parent_post_id)
                child_parent_identifier=rootID
#                 if not parent_node:
#                     print("Let's create %s"%parent_post_id)
#                     cascadet.create_node(parent_post_id, parent_post_id, parent=rootID,data=parent_node)
#                     parent_node=cascadet.get_node(parent_post_id)
                if parent_node:
                    child_parent_identifier=parent_node.identifier

                cascadet.create_node(comment_id, comment_id, parent=child_parent_identifier,data=child)
            except tree.DuplicatedNodeIdError as e:
                print("**",e)
                continue

        print(cascadet)
        return cascadet 
    
    def run_cascade_trees(self):
        self.cascade_trees=self.cascade_records[["rootID","rootUserID","nodeID","parentID","nodeUserID","short_propagation_delay","long_propagation_delay"]]
        self.cascade_trees["message"]=self.cascade_trees[["nodeID","parentID","nodeUserID","short_propagation_delay","long_propagation_delay"]].apply(lambda x: tuple(x),axis=1)
        self.cascade_trees=self.cascade_trees.groupby(['rootID','rootUserID'])["message"].apply(list).to_frame().reset_index()
        self.cascade_trees=self.cascade_trees[['rootID','rootUserID','message']].apply(self.get_cascade_tree,axis=1)
        np.save("%s/cascade_trees.npy"%(self.output_dir),self.cascade_trees)
        return self.cascade_trees
    
    def get_cascade_props(self,ctree):
        nodes=ctree.all_nodes()
        depth=ctree.depth()
        rid=ctree.root
        rnode=ctree.get_node(rid)
        #rnode_data=rnode.data
        #rauthor=rnode_data.get_node_author()
        for node in nodes:
            nid=node.identifier
            nlevel=ctree.level(nid)
            nchildren=ctree.children(nid)
            no_children=len(nchildren)

            parent=ctree.parent(nid)
            if(parent is not None):
                pid=parent.identifier
                ##pchildren=ctree.children(pid)
                ##p_no_children=len(pchildren)

                #pnode=ctree.get_node(pid)
                #pnode_data=pnode.data
                #pauthor=pnode_data.get_node_author()
            else:
                pid=-1
                #pauthor=-1
                ##p_no_children=-1


            #node_data=node.data

            #nauthor=node_data.get_node_author()

            #nshort_delay=node_data.get_node_short_delay()
            #nlong_delay=node_data.get_node_long_delay()

            #llist=[rid,rauthor,depth,nlevel,nid,nauthor,no_children,nshort_delay,nlong_delay,pid,pauthor]
            llist=[rid,depth,nlevel,nid,no_children,pid]
            
            ## only include non-leaves
            ##if(no_children!=0):
            self.cascade_props.append(llist)
        
    
    def run_cascade_props(self):
        for ctree in self.cascade_trees:
            ##self.get_cascade_props(ctree)
            self.pool.add_task(self.get_cascade_props,ctree)
        self.pool.wait_completion()
        #columns=["rootID","rootUserID","max_depth","level","nodeID","nodeUserID","degree","short_delay","long_delay","parentID","parentUserID"]
        columns=["rootID","max_depth","level","nodeID","degree","parentID"]
      
        self.cascade_props=pd.DataFrame(self.cascade_props,columns=columns)
        self.cascade_props.to_pickle("%s/cascade_props.pkl.gz"%(self.output_dir))
        return self.cascade_props
    
    def get_cascade_branching(self):
        cascade_props_degree=self.cascade_props.groupby(["level"])["degree"].apply(list).reset_index(name="degreeV")

        def _get_prob_vector(row):
            level=row['level']
            degree_list=row['degreeV']

            degree_bins = np.bincount(degree_list)
            degree_uniques = np.nonzero(degree_bins)[0]

            degree_matrix=np.vstack((degree_uniques,degree_bins[degree_uniques])).T

            degree_df=pd.DataFrame(degree_matrix,columns=["degree","count"])

            degree_df["probability"]=degree_df["count"]/degree_df["count"].sum()

            row['level']=level
            row['degreeV']=degree_list
            row['udegreeV']=degree_df['degree'].values
            row['probV']=degree_df["probability"].values

            return row

        cascade_props_degree=cascade_props_degree.apply(_get_prob_vector,axis=1)
        cascade_props_degree.set_index(["level"],inplace=True)
        cascade_props_degree.to_pickle("%s/cascade_props_prob_level_degree.pkl.gz"%(self.output_dir))
        return cascade_props_degree
    
#     def get_cascade_user_branching(self):
#         cascade_props_degree=self.cascade_props.groupby(["nodeUserID","level"])["degree"].apply(list).reset_index(name="degreeV")

#         def _get_prob_vector(row):
#             level=row['level']
#             degree_list=row['degreeV']

#             degree_bins = np.bincount(degree_list)
#             degree_uniques = np.nonzero(degree_bins)[0]

#             degree_matrix=np.vstack((degree_uniques,degree_bins[degree_uniques])).T

#             degree_df=pd.DataFrame(degree_matrix,columns=["degree","count"])

#             degree_df["probability"]=degree_df["count"]/degree_df["count"].sum()

#             row['level']=level
#             row['degreeV']=degree_list
#             row['udegreeV']=degree_df['degree'].values
#             row['probV']=degree_df["probability"].values

#             return row

#         cascade_props_degree=cascade_props_degree.apply(_get_prob_vector,axis=1)
#         cascade_props_degree.set_index(["level"],inplace=True)
#         cascade_props_degree.to_pickle("%s/cascade_props_prob_user_level_degree.pkl.gz"%(self.output_dir))
#         return cascade_props_degree
   
    
#     def get_cascade_delays(self):
#         cascade_props_size=self.cascade_props.groupby("rootID").size().reset_index(name="size")
#         cascade_props_delay=self.cascade_props.groupby("rootID")["long_delay"].apply(list).reset_index(name="delayV")
#         cascade_props_delay=pd.merge(cascade_props_delay,cascade_props_size,on="rootID",how="inner")
#         cascade_props_delay.to_pickle("%s/cascade_props_delay.pkl.gz"%(self.output_dir))
#         return cascade_props_delay


parser = argparse.ArgumentParser(description='Simulation Parameters')
parser.add_argument('--config', dest='config_file_path', type=argparse.FileType('r'))
args = parser.parse_args()

config_json=json.load(args.config_file_path)
platform = config_json['PLATFORM']
domain = config_json['DOMAIN']
scenario = config_json["SCENARIO"]

start_sim_period=config_json["START_SIM_PERIOD"]
end_sim_period=config_json["END_SIM_PERIOD"]
oneD=timedelta(days=1)
start_sim_period_date=datetime.strptime(start_sim_period,"%Y-%m-%d")
end_sim_period_date=datetime.strptime(end_sim_period,"%Y-%m-%d")
num_sim_days=(end_sim_period_date-start_sim_period_date).days+1

training_data_num_days=config_json["TRAINING_DATA_X_MUL_SIM"]
training_data_num_days=num_sim_days*training_data_num_days


train_start_date=start_sim_period_date-(timedelta(days=training_data_num_days))
###train_start_date=start_sim_period_date-(timedelta(days=training_data_num_days*2))
print("Train start date: ",train_start_date)
print("Train end date: ",start_sim_period_date)

num_training_days=(start_sim_period_date-train_start_date).days
print("# training days: %d"%num_training_days)
print("# simulation days: %d"%num_sim_days)



info_ids_path = config_json['INFORMATION_IDS']
info_ids_path = info_ids_path.format(platform)

### Load information IDs
info_ids = pd.read_csv(info_ids_path, header=None)
info_ids.columns = ['informationID']
info_ids = sorted(list(info_ids['informationID']))
#info_ids = ['informationID_'+x if 'informationID' not in x else x for x in info_ids]
print(len(info_ids),info_ids)


input_data_path = config_json["INPUT_CASCADES_FILE_PATH"]


try:
    cascade_records=pd.read_pickle(input_data_path)[["nodeID","parentID","rootID","nodeUserID","parentUserID","rootUserID","nodeTime","informationID"]]
    
#     for col in ['parentID','parentUserID','rootID','rootUserID','nodeID','nodeUserID']:
#         cascade_records.loc[cascade_records[col].isin(['?','not found','[Deleted]']),col]=None
        
    cascade_records['nodeTime']=pd.to_datetime(cascade_records['nodeTime'],infer_datetime_format=True)
    cascade_records.sort_values(by='nodeTime',inplace=True)
    cascade_records=cascade_records[cascade_records['nodeTime']<start_sim_period]
    cascade_records=cascade_records[cascade_records['nodeTime']>=train_start_date.strftime("%Y-%m-%d")]
    
    print("# Events: %d, # Messages: %d, # Info IDs: %d"%(cascade_records.shape[0],cascade_records['nodeID'].nunique(),cascade_records['informationID'].nunique()))
    
except KeyError:
    print("Reqd fields are missing in the input dataframe, they are nodeID,parentID,rootID,nodeUserID,parentUserID,rootUserID,nodeTime,informationID")
        

def _run(args):
    info_id=args[0]
    infoID_label=info_id.replace("/","_")
    print("InformationID: %s"%info_id)

    cascade_records_info=cascade_records.query('informationID==@info_id')
    ##cascade_records_info["isNew"]=~cascade_records_info['nodeUserID'].duplicated()
    ##cascade_records_info=cascade_records_info[cascade_records_info['nodeTime']>=train_start_date_newly.strftime("%Y-%m-%d")]

    cas=Cascade(platform,domain,scenario,info_id,cascade_records_info)
    cas.prepare_data()

    user_spread_info=cas.get_user_spread_info()
    print("saved, user spread info.")

    user_diffusion=cas.get_user_diffusion()
    print("saved, user diffusion probs.")

    cascade_trees=cas.run_cascade_trees()
    print("saved, cascade trees")
    cascade_props=cas.run_cascade_props()
    print("saved, cascade props")
    cascade_props_degree=cas.get_cascade_branching()
    print("saved, cascade level branching")
#     cascade_props_user_degree=cas.get_cascade_user_branching()
#     print("saved, cascade user, level branching")
#     cascade_props_delay=cas.get_cascade_delays()
#     print("saved, cascade delays")
    
for info_id in info_ids:#['arrests']:#['anti']:#
    _run([info_id])
    #break;
    
#Parallel(n_jobs=len(info_ids))(delayed(_run)([info_id]) for info_id in info_ids)