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
from libs.lib_job_thread import *

import logging

class SimX:
    def __init__(self,*args):
        self.platform=args[0]
        self.domain=args[1]
        self.scenario=args[2]
        self.model_identifier=args[3]
       
        self.pool=ThreadPool(32)
        self.sim_outputs=[]
        
        self.data_level_degree_list=None
        self.data_delay_level_degree_root_list=None
        
        self.data_user_list=None
        self.data_user_followers=None
        self.data_acts_list=None
        self.data_acts_list_indexed=None
        
        self.data_level_content_list=None

#         self.logger = logging.getLogger(__name__)
#         logPath='./logs/run_simulation_prob_HYDRA_%s_%s_S20001.log'%(self.platform,self.domain)
#         handler = logging.FileHandler(logPath,mode='w+')
#         handler.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         handler.setFormatter(formatter)
#         self.logger.addHandler(handler)
        
    def set_metadata(self):#,degree_list,delay_list):
        print("[Degree by level] loading..")     
        self.data_level_degree_list=pd.read_pickle("./metadata/probs/%s-%s/degree_cond_level.pkl.gz"%(self.platform,self.domain))

        print("[Delay sequences by size] loading..")
        self.data_delay_level_degree_root_list=pd.read_pickle("./metadata/probs/%s-%s/delay_cond_size.pkl.gz"%(self.platform,self.domain))



#         self.data_level_degree_list=degree_list
#         self.data_delay_level_degree_root_list=delay_list
        
        
    def set_user_metadata(self):#,user_list,user_followers):
        print("[User probability] loading..")
        self.data_user_list=pd.read_pickle("./metadata/probs/%s-%s/user_diffusion.pkl.gz"%(self.platform,self.domain))
        self.data_acts_list=self.data_user_list.groupby("pnodeUserID").sum()["no_responses"].reset_index(name="# acts")
        
#         print("[User followers] loading..")
#         self.data_user_followers=pd.read_pickle("./metadata/probs/%s-%s/user_followers.pkl.gz"%(self.platform,self.domain))
#         x=np.array(self.data_user_followers["user.followers_count"])
#         x_ranked=stats.rankdata(x, "average")/len(x)
#         self.data_user_followers['percentile']=x_ranked
        
        #self.data_acts_list_indexed=self.data_acts_list.set_index("pnodeUserID")
        self.data_user_list.set_index("pnodeUserID",inplace=True)
        
    def set_simulation_metadata(self,content_list):
        self.data_level_content_list=content_list
        
    def doSanity(self):
        # ## Given any level, return a node with degree X
        level=200
        b = self._get_degree(level)
        print("[sanity] Level: %d, Sampled Degree: %d"%(level,b))


        ## Given any size of the cascade, return a vector of delays
        size=10000
        dV = self._get_recorrected_delayV(size)
        print("[sanity] Expected: %d, Returned: %d Sampled Delay Vector: "%(size,dV.shape[0]),dV)

        # ## Given any degree in the first level, return an arbitrary cascade tree
        root_degree=3
        ctree=self._gen_cascade_tree(root_degree)
        print("[sanity] generated cascade tree")
        print(ctree)
        
    def _get_random_id(self):
        hash = random.getrandbits(64)
        return "%16x"%hash
    
    def _get_random_user_id(self):
        try:
            random_user_id=self.data_acts_list.sample(n=1,weights="# acts",replace=True).iloc[0]["pnodeUserID"]
        except KeyError as ke:
            random_user_id=self._get_random_id()
            ##print("new user: ",random_user_id)
            
        return random_user_id
    
    def _get_neighbor_user_id(self,user_id):
        try:
            ###random_user_id=self.data_user_list.loc[user_id].sample(n=1,weights="prob",replace=True).iloc[0]["source_author"]
            ##print(self.data_user_list[self.data_user_list['target_author']==user_id])
            neighbors=self.data_user_list.loc[user_id]#self.data_user_list[self.data_user_list['pnodeUserID']==user_id]
            if neighbors.shape[0]>0:
                random_user_id=neighbors.sample(n=1,weights="prob",replace=True).iloc[0]["nodeUserID"]
            else:
                random_user_id=self._get_random_user_id()
        except:
            random_user_id=self._get_random_user_id()
        return random_user_id

    def _get_random_users(self,size):
        return [self._get_random_id() for i in range(size)]
    
    def write_output(self,output,scenario,platform,domain,version):
        scenario=str(scenario)
        version=str(version)
        print("version %s"%version)
        #output_location="./output/%s/%s/%s/%s-%s_v%s.json"%(platform, domain, version, platform, domain, version)
        output_location="./output_%s/%s/%s/scenario_%s_domain_%s-%s_v%s.json"% (self.model_identifier,platform,domain,scenario,platform,domain,version)
        output_file = open(output_location, 'w', encoding='utf-8')
        output_records=output.to_dict('records')        
        for d in output_records:
            output_file.write(json.dumps(d) + '\n')
    
        
    def _get_degree(self,level,pa=False):
        sampled_degree=0
        ulevels=set(self.data_level_degree_list.index.get_level_values('level'))
        flag=False;
        while not flag:
            flag=(level in ulevels)
            if(flag==False):
                level-=1
          
        degreeList=np.array(self.data_level_degree_list.loc[level]['udegreeV'])
        degreeProbList=np.array(self.data_level_degree_list.loc[level]["probV"])
        if pa:
            degreeProbList=1/degreeProbList
            degreeProbList=degreeProbList/np.sum(degreeProbList)
        ##print(level,degreeList,degreeProbList)
        if len(degreeList)>0:
            sampled_degree = np.random.choice(a=degreeList, p=degreeProbList)
        return sampled_degree
    
#     def _get_pa_degree(self,level):
#         sampled_degree=0
#         ulevels=set(self.data_level_degree_list.index.get_level_values('level'))
#         flag=False;
#         while not flag:
#             flag=(level in ulevels)
#             if(flag==False):
#                 level-=1
        
        
#         degreeList=np.array(self.data_level_degree_list.loc[level]['udegreeV'])
#         degreeProbList=np.array(1-self.data_level_degree_list.loc[level]["probV"])
        
#         if len(degreeList)>0:
#             sampled_degree = np.random.choice(a=degreeList, p=degreeProbList)
#         return sampled_degree

    def _get_delayV(self,size):
        sample_delays=self.data_delay_level_degree_root_list[self.data_delay_level_degree_root_list["size"]==size]
        no_records=sample_delays.shape[0]

        if no_records>0:
            sample_delay=sample_delays.sample(n=1, replace=False)
            dV=np.array(list(sample_delay["delayV"]))[0]

            return dV

        else:
            max_size=self.data_delay_level_degree_root_list["size"].max()
            if(size>max_size):
                return self._get_delayV(max_size)
            else:
                return self._get_delayV(size+1)

    def _get_recorrected_delayV(self,size):
        dV = self._get_delayV(size)
        if(dV.shape[0]>size):
            dV=dV[:size]
        else:
            max_ldelay=np.max(dV)
            for n in range(len(dV), size):
                dV=np.append(dV,max_ldelay)
        return dV
    
#     def _get_contentV(self,level):
#         ulevels=set(self.data_level_content_list.index.get_level_values('level'))
#         flag=False;
#         while not flag:
#             flag=(level in ulevels)
#             if(flag==False):
#                 level-=1
#         contentV=self.data_level_content_list.iloc[level]['contentV']
#         sampled_contentV = contentV[np.random.randint(0,len(contentV))]
#         return sampled_contentV[1:]


    def _get_synthetic_tree_recursive(self,level,pdegree,cascade_tree_matrix,nlist):

        if(cascade_tree_matrix is None):
            cascade_tree_matrix=[]
            cascade_tree_matrix.append(nlist)

        children=pdegree

        while(children>0):
            mid=self._get_random_id()
            pid=nlist[2]
            puser_id=nlist[4]
            
            
            nuser_id=self._get_neighbor_user_id(puser_id)
            ###nuser_id=self._get_random_id()
  
            ndegree=self._get_degree(level)
            
            klist=[level,ndegree,mid,pid,nuser_id]

            cascade_tree_matrix.append(klist)
            self._get_synthetic_tree_recursive(level+1,ndegree,cascade_tree_matrix,klist)
            children-=1

        return cascade_tree_matrix

    def _gen_cascade_tree(self,pid=None,puser_id=None,pdegree=None):
        level=0
        
        ## post id
        if pid is None:
            pid=self._get_random_id()
        ## post user id 
        if puser_id is None:
            puser_id=self._get_random_user_id()
        ## post degree
        if pdegree is None:
            pdegree=self._get_degree(level)
            
        ## level, my degree, my id, my parent id
        nlist=[level,pdegree,pid,pid,puser_id]
        cascade_tree_matrix=self._get_synthetic_tree_recursive(level+1,pdegree,None,nlist)
        cascade_tree=pd.DataFrame(cascade_tree_matrix,columns=["level","degree","nodeID","parentID","nodeUserID"])
        ##print(cascade_tree.shape[0])
        cascade_tree["rootID"]=pid
        cascade_tree["actionType"]="retweet"  #'retweet
        cascade_tree.loc[:0,"actionType"] ="tweet"

        ## attach the delays
        ctree_size=cascade_tree.shape[0]
        cascade_tree["long_propagation_delay"]=self._get_recorrected_delayV(ctree_size)
        return cascade_tree


    def _simulate(self,ipost):
        
        ipost_id=ipost['id_h']
        ipost_user=ipost['author_h']
        ipost_degree=None
            

        ipost_created_date=str(ipost['created_date'])
        ##ipost_subreddit=ipost['subreddit_id']

        ##print("started: ",start)
        ##print("[simulation] post id: %s, author: %s, timestamp: %s"%(ipost_id,ipost_user,ipost_created_date))

        ipost_tree=self._gen_cascade_tree(ipost_id,ipost_user,ipost_degree)
        ##print("-",ipost_tree)
        # change the post id
#         assigned_rootID=ipost_tree['rootID'].iloc[0]
#         ipost_tree.replace({assigned_rootID: ipost_id}, regex=True,inplace=True)

        # assign times
        ipost_tree["nodeTime"]=ipost_created_date
        ipost_tree["nodeTime"]=pd.to_datetime(ipost_tree["nodeTime"])
        ipost_tree["nodeTime"]+=ipost_tree["long_propagation_delay"]

#         # assign authors
#         ipost_tree_size=ipost_tree.shape[0]
#         ## random users
#         ipost_tree["nodeUserID"]=self._get_random_users(ipost_tree_size)
#         ## fix the poster
#         ipost_tree.loc[:0,"nodeUserID"]=ipost_user


        icols=["nodeID","nodeUserID","parentID", "rootID", "actionType", "nodeTime"]
        ipost_tree=ipost_tree[icols]

        ## change to timestamp
        ipost_tree["nodeTime"]=ipost_tree["nodeTime"].values.astype(np.int64) // 10 ** 9
        
        print("[simulation] post id: %s, author: %s, timestamp: %s, cascade size: %d"%(ipost_id,ipost_user,ipost_created_date,ipost_tree.shape[0]))

#         ## assign communityID
#         ipost_tree["communityID"]=ipost_subreddit
        ##print("--",ipost_tree)
        self.sim_outputs.append(ipost_tree)
        return ipost_tree
    
    def _run_simulate(self,iposts_records,version):
        start = time.time()
        for ipost in iposts_records:
            ##self._simulate(ipost)
            self.pool.add_task(self._simulate,ipost)
        self.pool.wait_completion()
        end = time.time()
        elapsed=end - start
        
        
        sim_output=pd.concat(self.sim_outputs)
        
        no_cascades=len(self.sim_outputs)
        no_acts=sim_output.shape[0]
        print("[simulation completed] version: %s, # cascades: %d,%d, # acts: %d, Elapsed %.3f seconds."%(version, no_cascades,sim_output['rootID'].nunique(),no_acts,elapsed))


        self.write_output(sim_output,scenario=self.scenario,platform=self.platform,domain=self.domain,version=version)
        