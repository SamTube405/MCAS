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
       
        #self.pool=ThreadPool(64)
        ##self.sim_outputs=[]
        self.output_location="./output/%s/%s/%s/%s"% (self.platform,self.domain,self.scenario,self.model_identifier)
        print("[output dir] %s"%self.output_location)
        
    def set_metadata(self):
        print("[Degree by level] loading..")
        self.data_level_degree_list=pd.read_pickle("./metadata/probs/%s/%s/%s/%s/cascade_props_prob_level_degree.pkl.gz"%(self.platform,self.domain,self.scenario,self.infoID_label))
#         print("[Degree by user, level] loading..")
#         self.data_user_level_degree_list=pd.read_pickle("./metadata/probs/%s/%s/%s/%s/cascade_props_prob_user_level_degree.pkl.gz"%(self.platform,self.domain,self.scenario,self.infoID_label))
        
        
#         print("[Delay sequences by size] loading..")
#         self.data_delay_level_degree_root_list=pd.read_pickle("./metadata/probs/%s-%s/delay_cond_size.pkl.gz"%(self.platform,self.domain))
    
        
    def set_user_metadata(self):#,user_list,user_followers):
        print("[User probability] loading..")
        self.data_user_list=pd.read_pickle("./metadata/probs/%s/%s/%s/%s/user_diffusion.pkl.gz"%(self.platform,self.domain,self.scenario,self.infoID_label))
        ##self.data_user_list=self.data_user_list[self.data_user_list['total_num_responses']>1]
        self.data_user_ego=self.data_user_list.groupby("parentUserID").size().reset_index(name="num_neighbors")
        self.data_user_ego.set_index("parentUserID",inplace=True)
        
        
#         self.data_parent_acts_list=self.data_user_list.groupby("parentUserID").sum()["num_responses"].reset_index(name="num_acts")
#         self.data_parent_acts_list.set_index("parentUserID",inplace=True)
        
        self.data_acts_list=self.data_user_list.groupby("nodeUserID").sum()["num_responses"].reset_index(name="num_acts")
        self.data_acts_list.set_index("nodeUserID",inplace=True)
        
#         self.data_reshare_list=self.data_user_list.groupby("nodeUserID").size().reset_index(name="num_parents")
#         self.data_reshare_list['attention']=1/self.data_reshare_list["num_parents"]
        ##self.data_reshare_list.set_index("nodeUserID",inplace=True)
        
        self.data_user_list.set_index("parentUserID",inplace=True)
        
        self.data_spread_score=pd.read_pickle("./metadata/probs/%s/%s/%s/%s/user_spread_info.pkl.gz"%(self.platform,self.domain,self.scenario,self.infoID_label))
        self.data_seed_score=self.data_spread_score[self.data_spread_score['num_seeds']>0]
        self.data_response_score=self.data_spread_score[self.data_spread_score['num_responses']>0]
        if self.data_response_score.shape[0]==0:
            self.data_response_score=self.data_spread_score
            self.data_response_score['num_responses']=1
        
        ##self.data_user_ego=pd.merge(self.data_user_ego,self.data_spread_score,left_index=True,right_index=True,how='inner')
        ##self.data_user_ego=self.data_user_ego.query('spread_score>0')
        
#     def set_simulation_metadata(self,content_list):
#         self.data_level_content_list=content_list
        
#     def doSanity(self):
#         # ## Given any level, return a node with degree X
#         level=200
#         b = self._get_degree(level)
#         print("[sanity] Level: %d, Sampled Degree: %d"%(level,b))


#         ## Given any size of the cascade, return a vector of delays
#         size=10000
#         dV = self._get_recorrected_delayV(size)
#         print("[sanity] Expected: %d, Returned: %d Sampled Delay Vector: "%(size,dV.shape[0]),dV)

#         # ## Given any degree in the first level, return an arbitrary cascade tree
#         root_degree=3
#         ctree=self._gen_cascade_tree(root_degree)
#         print("[sanity] generated cascade tree")
#         print(ctree)
        
    def _get_random_id(self):
        hash = random.getrandbits(64)
        return "%16x"%hash
    
#     def _get_random_user_id(self):
#         hash = random.getrandbits(64)
#         return "gen_%16x"%hash

    def _get_activity_biased_seed_user_id(self):
        random_user_id=self.data_seed_score.sample(n=1,weights="num_seeds").index[0]
            
        return random_user_id
    
    def _get_activity_biased_response_user_id(self):
        random_user_id=self.data_response_score.sample(n=1,weights="num_responses").index[0]
            
        return random_user_id
    
    def _get_activity_biased_response_user_ids(self,num_users):
        random_user_ids=list(self.data_response_score.sample(n=num_users,weights="num_responses",replace=False)['nodeUserID'])
            
        return random_user_ids
    
    def _get_ego_biased_user_id(self,degree):
#         random_user_id=self.data_spread_score.sample(n=1,weights="num_seeds").index[0]
        hops=self.data_user_ego.query('num_neighbors>=@degree')
    
        if hops.shape[0]>0:
            random_user_id=hops.sample(n=1).index[0]#weights=weights="num_neighbors"
            #print('Ego biased seed user assigned, id: %s, degree: %d'%(random_user_id,degree))
        else:
            random_user_id=self.data_user_ego.sample(n=1,weights="num_neighbors").index[0]
   
            
        return random_user_id
    
#     def _prioritize_neighbor_user_ids(self,neighbor_ids):
#         listed_neighbor_ids=neighbor_ids
#         hop_neighbors=self.data_user_ego[self.data_user_ego.index.isin(neighbor_ids)]
#         if hop_neighbors.shape[0]>0:
#             hop_neighbors.sort_values(by="num_neighbors",ascending=False)
#             listed_neighbor_ids=set(hop_neighbors.index)
            
#             missed_neighbor_ids=set(neighbor_ids)-listed_neighbor_ids
#             listed_neighbor_ids=list(listed_neighbor_ids)
#             if len(missed_neighbor_ids)>0:
#                 act_neighbors=self.data_acts_list[self.data_acts_list.index.isin(missed_neighbor_ids)]
#                 act_neighbors.sort_values(by="num_acts",ascending=False)
#                 listed_neighbor_ids_=set(act_neighbors.index)
#                 missed_neighbor_ids_=set(missed_neighbor_ids)-listed_neighbor_ids_
#                 listed_neighbor_ids_=list(listed_neighbor_ids_)
#                 missed_neighbor_ids_=list(missed_neighbor_ids_)
                
#                 if len(missed_neighbor_ids_)>0:
#                     listed_neighbor_ids_.extend(missed_neighbor_ids_)
            
#                 listed_neighbor_ids.extend(listed_neighbor_ids_)
#         else:
#             ##print(neighbor_ids)
#             act_neighbors=self.data_acts_list[self.data_acts_list.index.isin(neighbor_ids)]
#             act_neighbors.sort_values(by="num_acts",ascending=False)
            
#             if act_neighbors.shape[0]>0:
#                 listed_neighbor_ids=set(act_neighbors.index)
#                 missed_neighbor_ids=set(neighbor_ids)-listed_neighbor_ids
#                 listed_neighbor_ids=list(listed_neighbor_ids)
#                 missed_neighbor_ids=list(missed_neighbor_ids)
#                 if len(missed_neighbor_ids)>0:
#                     listed_neighbor_ids.extend(missed_neighbor_ids)
    
#         assert(len(neighbor_ids)==len(listed_neighbor_ids))
#         return listed_neighbor_ids
        
    
    def _get_neighbor_user_ids(self,user_id,degree):
        try:
            neighbors=self.data_user_list.loc[[user_id]]
            num_neighbors=neighbors.shape[0]

            neighbor_user_ids=list(neighbors.sample(n=degree,weights="prob",replace=True)['nodeUserID'])
                
        except KeyError as ke:
            neighbor_user_ids=list(self.data_acts_list.sample(n=degree,weights="num_acts",replace=True).index)
         
        assert(degree==len(neighbor_user_ids))
        return neighbor_user_ids
    
#     def _get_neighbor_user_id(self,user_id):
#         try:
#             neighbors=self.data_user_list.loc[[user_id]]

#             neighbor_user_id=neighbors.sample(n=1,weights="prob",replace=True).iloc[0]['nodeUserID']
                
#         except KeyError as ke:
#             neighbor_user_id=self._get_activity_biased_user_id()
         
#         return neighbor_user_id

    

    
    def write_output(self,output,version):
        output_loc="%s/cascade_v%s.pkl.gz"% (self.output_location,version)
        output.to_pickle(output_loc)
        
#         output_file = open(output_loc, 'w', encoding='utf-8')
#         output_records=output.to_dict('records')        
#         for d in output_records:
#             output_file.write(json.dumps(d) + '\n')
    
        
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
    

#     def _get_delayV(self,size):
#         sample_delays=self.data_delay_level_degree_root_list[self.data_delay_level_degree_root_list["size"]==size]
#         no_records=sample_delays.shape[0]

#         if no_records>0:
#             sample_delay=sample_delays.sample(n=1, replace=False)
#             dV=np.array(list(sample_delay["delayV"]))[0]

#             return dV

#         else:
#             max_size=self.data_delay_level_degree_root_list["size"].max()
#             if(size>max_size):
#                 return self._get_delayV(max_size)
#             else:
#                 return self._get_delayV(size+1)

#     def _get_recorrected_delayV(self,size):
#         dV = self._get_delayV(size)
#         if(dV.shape[0]>size):
#             dV=dV[:size]
#         else:
#             max_ldelay=np.max(dV)
#             for n in range(len(dV), size):
#                 dV=np.append(dV,max_ldelay)
#         return dV
    
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

        pid=nlist[2]
        puser_id=nlist[4]
        num_children=pdegree

        
        ndegrees=self._get_degree_vector(level,num_children)
        #ndegrees=[self._get_degree(level) for _ in range(num_children)]
#         if self.neighborhood_attention:
#             nuser_ids=self._get_neighbor_user_ids(puser_id,num_children)
#         else:
        nuser_ids=[]
        for ndegree in ndegrees:
            if ndegree>0:
                nuser_id=self._get_ego_biased_user_id(ndegree)
            else:
                nuser_id=self._get_activity_biased_response_user_id()
            nuser_ids.append(nuser_id)


        assert(num_children==len(nuser_ids))

        
        index=0
        while(index<num_children):
            mid=self._get_random_id()
            
            ##nuser_id=self._get_neighbor_user_id(puser_id)
            ###nuser_id=self._get_random_id()
            #nuser_id=nuser_ids[index]
  
            ndegree=ndegrees[index]#self._get_degree(level)
            nuser_id=nuser_ids[index]
            ##nuser_id=self._get_ego_hop_biased_user_id(puser_id,ndegree)
            
            klist=[level,ndegree,mid,pid,nuser_id,puser_id]

            cascade_tree_matrix.append(klist)
            self._get_synthetic_tree_recursive(level+1,ndegree,cascade_tree_matrix,klist)
            index+=1

        return cascade_tree_matrix

    def _gen_cascade_tree(self,pid=None,puser_id=None,pdegree=None):
        level=0
        
        ## post id
        if pid is None:
            pid=self._get_random_id()
        ## post degree
        if pdegree is None:
            pdegree=self._get_degree(level)
        ## post user id 
        if puser_id is None:
            puser_id=self._get_ego_biased_user_id(pdegree)
            ##--puser_id=self._get_activity_biased_seed_user_id()

            ##--pdegree=self._get_degree(puser_id,level)
            
        
        
            
        ## level, my degree, my id, my parent id
        nlist=[level,pdegree,pid,pid,puser_id,puser_id]
        if pdegree>0:
            cascade_tree_matrix=self._get_synthetic_tree_recursive(level+1,pdegree,None,nlist)
        else:
            cascade_tree_matrix=[nlist]
        cascade_tree=pd.DataFrame(cascade_tree_matrix,columns=["level","degree","nodeID","parentID","nodeUserID","parentUserID"])
        ##print(cascade_tree.shape[0])
        cascade_tree["rootID"]=pid
        cascade_tree["rootUserID"]=puser_id
        cascade_tree["actionType"]=self.response_label
        cascade_tree.loc[:0,"actionType"] =self.seed_label

        ## attach the delays
        ctree_size=cascade_tree.shape[0]
#         cascade_tree["long_propagation_delay"]=self._get_recorrected_delayV(ctree_size)
        return cascade_tree


    def _simulate(self,ipost):
        
        ipost_id=ipost['nodeID']
        ipost_user=None#ipost['nodeUserID']
        ipost_degree=None#ipost['iDegree']##None
        ipost_created_date=str(ipost['nodeTime'])
        ipost_infoID=self.infoID#ipost['informationID']
                    
        ipost_tree=self._gen_cascade_tree(ipost_id,ipost_user,ipost_degree)

        # assign times
        ipost_tree["nodeTime"]=ipost_created_date
        ipost_tree["nodeTime"]=pd.to_datetime(ipost_tree["nodeTime"])
#         ipost_tree["nodeTime"]+=ipost_tree["long_propagation_delay"]
        
        ipost_tree["informationID"]=ipost_infoID


        icols=["nodeID","nodeUserID","parentID", "parentUserID", "rootID", "rootUserID", "actionType", "nodeTime","informationID"]
        ipost_tree=ipost_tree[icols]

        ## change to timestamp
        ipost_tree["nodeTime"]=ipost_tree["nodeTime"].values.astype(np.int64) // 10 ** 9
        
        ipost_user=ipost_tree.iloc[0]['rootUserID']
        
        #ipost_tree.loc[ipost_tree['nodeUserID']=="new_",'nodeUserID']="new_"+ipost_tree['nodeUserID'].apply(lambda x: self._get_random_id())
        
        print("[simulation] infoID: %s, post id: %s, author: %s, timestamp: %s, cascade size: %d"%(ipost_infoID,ipost_id,ipost_user,ipost_created_date,ipost_tree.shape[0]))

#         ## assign communityID
#         ipost_tree["communityID"]=ipost_subreddit
        ##print("--",ipost_tree)
        ##self.sim_outputs.append(ipost_tree)
        return ipost_tree
    
    def _run_simulate(self,iposts_records,version):
        start = time.time()
        
        
        num_shares=int(iposts_records[0])
        num_ousers=int(iposts_records[1])
        num_nusers=int(iposts_records[2])
        num_shares=max(num_shares,num_ousers+num_nusers)
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
        #print("***",version,sim_output.shape[0],int(num_shares))

        sim_output=sim_output.sample(n=num_shares)
        assert(sim_output.shape[0]==num_shares)
         
        end = time.time()
        elapsed=end - start
        print("[simulation completed] version: %s, # cascades: %d,%d, # acts: %d, Elapsed %.3f seconds."%(version, no_cascades,sim_output['rootID'].nunique(),no_acts,elapsed))


        self.write_output(sim_output,version=version)
        ##self.sim_outputs=[]
        