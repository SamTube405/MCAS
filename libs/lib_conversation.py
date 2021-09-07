import numpy as np
import pandas as pd
import sys,os
#from random import choices
import random
from datetime import datetime as dt
import json
from ast import literal_eval
import time
from scipy import stats

#from joblib import Parallel, delayed
#from libs.lib_job_thread import *

import logging
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class SimX:
    def __init__(self,*args):
        self.platform=args[0]
        self.domain=args[1]
        self.scenario=args[2]
        self.model_identifier=args[3]
        self.infoID=args[4]
        self.infoID_label=self.infoID.replace("/","_")
        
        if self.platform=="twitter":
            self.seed_label='tweet'
            self.response_label='retweet'
        elif self.platform=="youtube":
            self.seed_label='video'
            self.response_label='comment'
        elif self.platform=="reddit":
            self.seed_label='post'
            self.response_label='comment'
        elif self.platform=="jamii":
            self.seed_label='post'
            self.response_label='comment'
       
        self.output_location="./output/%s/%s/%s/%s"% (self.platform,self.domain,self.scenario,self.model_identifier)
        print("[output dir] %s"%self.output_location)
        
    def set_metadata(self):
        print("[Degree by level] loading..")
        self.data_level_degree_list=pd.read_pickle("./metadata/probs/%s/%s/%s/%s/cascade_props_prob_level_degree.pkl.gz"%(self.platform,self.domain,self.scenario,self.infoID_label))
    
        
    def set_user_metadata(self):#,user_list,user_followers):
        print("[User probability] loading..")
        self.data_user_list=pd.read_pickle("./metadata/probs/%s/%s/%s/%s/user_diffusion.pkl.gz"%(self.platform,self.domain,self.scenario,self.infoID_label))
        ##self.data_user_list=self.data_user_list[self.data_user_list['total_num_responses']>1]
        self.data_user_ego=self.data_user_list.groupby("parentUserID").size().reset_index(name="num_neighbors")
        self.data_user_ego.set_index("parentUserID",inplace=True)

        
        self.data_spread_score=pd.read_pickle("./metadata/probs/%s/%s/%s/%s/user_spread_info.pkl.gz"%(self.platform,self.domain,self.scenario,self.infoID_label))
        self.data_seed_score=self.data_spread_score[self.data_spread_score['num_seeds']>0]
        self.data_response_score=self.data_spread_score[self.data_spread_score['num_responses']>0]
        if self.data_response_score.shape[0]==0:
            self.data_response_score=self.data_spread_score
            self.data_response_score['num_responses']=1
 
        
    def _get_random_id(self):
        hash = random.getrandbits(64)
        return "%16x"%hash
  

    
    def write_output(self,output,version):
        output_loc="%s/cascade_v%s.pkl.gz"% (self.output_location,version)
        output.to_pickle(output_loc)
        
    def _get_degree(self,level):
        sampled_degree=0
        ulevels=set(self.data_level_degree_list.index.get_level_values('level'))
        flag=False;
        while not flag:
            flag=(level in ulevels)
            if(flag==False):
                level-=1
          
        degreeList=np.array(self.data_level_degree_list.loc[level]['udegreeV'])
        degreeProbList=np.array(self.data_level_degree_list.loc[level]["probV"])

        if len(degreeList)>0:
            sampled_degree = np.random.choice(a=degreeList, p=degreeProbList)
        return sampled_degree

    
    def _get_degree_vector(self,level,num_children):
        sampled_degrees=np.zeros(num_children)
        ulevels=set(self.data_level_degree_list.index.get_level_values('level'))
        flag=False;
        while not flag:
            flag=(level in ulevels)
            if(flag==False):
                level-=1
          
        degreeList=np.array(self.data_level_degree_list.loc[level]['udegreeV'])
        degreeProbList=np.array(self.data_level_degree_list.loc[level]["probV"])

        if len(degreeList)>0:
            ## sort desc. order
            sampled_degrees = -np.sort(-np.random.choice(size=num_children,a=degreeList, p=degreeProbList))

        assert(num_children==len(sampled_degrees))
        return sampled_degrees
    


    def _get_synthetic_tree_recursive(self,level,pdegree,cascade_tree_matrix,nlist):

        if(cascade_tree_matrix is None):
            cascade_tree_matrix=[]
            cascade_tree_matrix.append(nlist)

        pid=nlist[2]
        num_children=pdegree

        ndegrees=self._get_degree_vector(level,num_children)

        index=0
        while(index<num_children):
            mid=self._get_random_id()
  
            ndegree=ndegrees[index]
            
            klist=[level,ndegree,mid,pid]

            cascade_tree_matrix.append(klist)
            self._get_synthetic_tree_recursive(level+1,ndegree,cascade_tree_matrix,klist)
            index+=1

        return cascade_tree_matrix

    def _gen_cascade_tree(self,pid=None,pdegree=None):
        level=0
        
        ## post id
        if pid is None:
            pid=self._get_random_id()
        ## post degree
        if pdegree is None:
            pdegree=self._get_degree(level)
        
        
            
        ## level, my degree, my id, my parent id
        nlist=[level,pdegree,pid,pid]
        if pdegree>0:
            cascade_tree_matrix=self._get_synthetic_tree_recursive(level+1,pdegree,None,nlist)
        else:
            cascade_tree_matrix=[nlist]
        cascade_tree=pd.DataFrame(cascade_tree_matrix,columns=["level","degree","nodeID","parentID"])
        ##print(cascade_tree.shape[0])
        cascade_tree["rootID"]=pid
        cascade_tree["actionType"]=self.response_label
        cascade_tree.loc[:0,"actionType"] =self.seed_label

        ## attach the delays
        ctree_size=cascade_tree.shape[0]
#         cascade_tree["long_propagation_delay"]=self._get_recorrected_delayV(ctree_size)
        return cascade_tree


    def _simulate(self,ipost):
        
        ipost_id=ipost['nodeID']
        ipost_degree=None
        ipost_created_date=str(ipost['nodeTime'])
        ipost_infoID=self.infoID
                    
        ipost_tree=self._gen_cascade_tree(ipost_id,ipost_degree)

        # assign times
        ipost_tree["nodeTime"]=ipost_created_date
        ipost_tree["nodeTime"]=pd.to_datetime(ipost_tree["nodeTime"])
        
        ipost_tree["informationID"]=ipost_infoID


        icols=["nodeID","parentID","rootID", "actionType", "nodeTime","informationID"]
        ipost_tree=ipost_tree[icols]

        ## change to timestamp
        ipost_tree["nodeTime"]=ipost_tree["nodeTime"].values.astype(np.int64) // 10 ** 9
        

        
        print("[simulation] infoID: %s, post id: %s, timestamp: %s, cascade size: %d"%(ipost_infoID,ipost_id,ipost_created_date,ipost_tree.shape[0]))


        return ipost_tree
    
    def _run_simulate(self,iposts_records,version):
        start = time.time()
        
        
        num_shares=int(iposts_records[0])
        num_ousers=int(iposts_records[1])
        num_nusers=int(iposts_records[2])
        num_shares=int(max(num_shares,num_ousers+num_nusers))
        iposts_time=iposts_records[3]
        
        if num_shares==0:
            return
            
        sim_outputs=[]
        issued=num_shares
        while issued>0:
            ipost={}
            ipost['nodeID']="seed_"+self._get_random_id()#
            ipost['nodeTime']=iposts_time
            ipost_tree=self._simulate(ipost)
            sim_outputs.append(ipost_tree)
            #print("cascade size: %d"%ipost_tree.shape[0])      
            issued-=ipost_tree.shape[0]

        
        sim_output=pd.concat(sim_outputs)
        sim_output['platform']=self.platform
        
        no_cascades=len(sim_outputs)
        no_acts=sim_output.shape[0]

        sim_output=sim_output.sample(n=num_shares)
        assert(sim_output.shape[0]==num_shares)
        
#         ## Fixing for old users
#         response_score=self.data_response_score.sample(n=num_ousers,weights="num_responses",replace=True)
        
        ## For broken conversations Only
        sim_output['rootUserID']=self.data_user_ego.sample(n=num_shares,weights="num_neighbors",replace=True).index
        sim_output['parentUserID']=self.data_user_ego.sample(n=num_shares,weights="num_neighbors",replace=True).index
        
        ##sim_output['nodeUserID']=self.data_response_score.sample(n=num_shares,weights="num_responses",replace=True).index
        
        sim_outputs=[]
        parent_acts=sim_output.groupby('parentUserID').size()
        for parent in parent_acts.index:
            num_parent_responses=parent_acts[parent]
            sim_output_parent=sim_output.query('parentUserID==@parent')
            sim_output_parent['nodeUserID']=self.data_response_score.sample(n=num_parent_responses,weights="num_responses",replace=True).index
            sim_outputs.append(sim_output_parent)
        sim_output=pd.concat(sim_outputs)

        
        
        end = time.time()
        elapsed=end - start
        print("[simulation completed] version: %s, # cascades: %d,%d, # acts: %d, Elapsed %.3f seconds."%(version, no_cascades,sim_output['rootID'].nunique(),no_acts,elapsed))


        self.write_output(sim_output,version=version)
        