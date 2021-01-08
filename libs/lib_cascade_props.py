import os
import pandas as pd
import numpy as np
from treelib import Node, Tree, tree
from lib_job_thread import *

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
    def __init__(self,domain,platform,data_path):
        self.domain=domain
        self.platform=platform
        self.output_dir="./metadata/probs/%s-%s"%(self.platform,self.domain)
        create_dir(self.output_dir)
        
        self.pool = ThreadPool(128) 
        self.cascade_props=[]

        
        
        try:
            ## Expected: "nodeID","parentID","rootID","nodeUserID","nodeTime"
            self.cascade_records=pd.read_pickle(data_path)[["nodeID","parentID","rootID","nodeUserID","nodeTime"]]
            self.cascade_records=self.cascade_records.drop_duplicates()
            self.cascade_records['actionType']='response'
            self.cascade_records.loc[self.cascade_records["nodeID"]==self.cascade_records["rootID"],"actionType"]="seed"
            print("# cascades: %d, # nodes: %d"%(self.cascade_records['rootID'].nunique(),self.cascade_records.shape[0]))
        except KeyError:
            print("Reqd fields are missing in the input dataframe, they are nodeID,parentID,rootID,nodeUserID,nodeTime")
        
    def prepare_data(self):
        node_users=self.cascade_records[['nodeID','nodeUserID','nodeTime']].drop_duplicates()
        
        node_users.columns=['rootID','rootUserID','rootTime']
        self.cascade_records=pd.merge(self.cascade_records,node_users,on='rootID',how='inner')
        
        node_users.columns=['parentID','parentUserID','parentTime']
        self.cascade_records=pd.merge(self.cascade_records,node_users,on='parentID',how='inner')
        
        self.cascade_records["short_propagation_delay"]=self.cascade_records['nodeTime']-self.cascade_records['parentTime']
        self.cascade_records["long_propagation_delay"]=self.cascade_records['nodeTime']-self.cascade_records['rootTime']
        self.cascade_records.to_pickle("%s/cascade_records.pkl.gz"%(self.output_dir))
        
    def get_user_diffusion(self):
        user_diffusion=self.cascade_records.groupby(['parentUserID','nodeUserID']).size().reset_index(name='num_responses')
        user_diffusion_=self.cascade_records.groupby(['parentUserID']).size().reset_index(name='total_num_responses')
        user_diffusion=pd.merge(user_diffusion,user_diffusion_,on='parentUserID',how='inner')
        user_diffusion['prob']=user_diffusion['num_responses']/user_diffusion['total_num_responses']
        user_diffusion.to_pickle("%s/user_diffusion.pkl.gz"%(self.output_dir))
        return user_diffusion
    
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
        
        cuts=np.arange(50,num_seed_users,50)
        for i in cuts:
            x=self.spread_info.iloc[:i]['num_responses_recvd'].sum()/self.spread_info['num_responses_recvd'].sum()
            #print("%0.2f responses covered by Top-%d influential users."%(x,i))
            if x>0.9:
                print("%0.2f responses covered by Top-%d influential users."%(x,i))
                top_k=i
                break;
        self.top_k_influentials=list(self.spread_info.iloc[0:top_k].index)
        np.save("%s/top_k_influentials.npy"%(self.output_dir),self.top_k_influentials)
                
        self.spread_info.to_pickle("%s/user_spread_info.pkl.gz"%(self.output_dir))
        return self.spread_info
       


    def get_cascade_tree(self,cascade_tuple):
        rootID=cascade_tuple[0]
        rootUserID=cascade_tuple[1]
        childNodes=cascade_tuple[2]
        
        cascadet=Tree()

        ##print(post_id,author)
        parent=Node(rootID,rootUserID,0,0)
        cascadet.create_node(rootID, rootID, data=parent)

        for m in childNodes:
            comment_id=m[0]
            parent_post_id=m[1]
            child_author_id=m[2]
            short_delay=m[3]
            long_delay=m[4]
            child=Node(comment_id,child_author_id,short_delay,long_delay)

            try:
                parent_node=cascadet.get_node(parent_post_id)
                ##print("COMMENT: ",comment_id)

                try:
                    cascadet.create_node(comment_id, comment_id, parent=parent_node.identifier,data=child)
                except (tree.DuplicatedNodeIdError,AttributeError) as e:
                    ##print(e)
                    continue
            except KeyError as ke:
                ##print(ke)
                continue;

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
        rnode_data=rnode.data
        rauthor=rnode_data.get_node_author()
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

                pnode=ctree.get_node(pid)
                pnode_data=pnode.data
                pauthor=pnode_data.get_node_author()
            else:
                pid=-1
                pauthor=-1
                ##p_no_children=-1


            node_data=node.data

            nauthor=node_data.get_node_author()

            nshort_delay=node_data.get_node_short_delay()
            nlong_delay=node_data.get_node_long_delay()

            llist=[rid,rauthor,depth,nlevel,nid,nauthor,no_children,nshort_delay,nlong_delay,pid,pauthor]
            ## only include non-leaves
            ##if(no_children!=0):
            self.cascade_props.append(llist)
        
    
    def run_cascade_props(self):
        for ctree in self.cascade_trees:
            ##self.get_cascade_props(ctree)
            self.pool.add_task(self.get_cascade_props,ctree)
        self.pool.wait_completion()
        columns=["rootID","rootUserID","max_depth","level","nodeID","nodeUserID","degree","short_delay","long_delay","parentID","parentUserID"]
        self.cascade_props=pd.DataFrame(self.cascade_props,columns=columns)
        
        self.cascade_props['inf_rootID']=False
        self.cascade_props.loc[self.cascade_props['rootUserID'].isin(self.top_k_influentials)==True,"inf_rootID"]=True
        
        self.cascade_props.to_pickle("%s/cascade_props.pkl.gz"%(self.output_dir))
        return self.cascade_props
    
    def get_cascade_branching(self):
        cascade_props_degree=self.cascade_props.groupby("level")["degree"].apply(list).reset_index(name="degreeV")

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
        cascade_props_degree.set_index('level',inplace=True)
        cascade_props_degree.to_pickle("%s/cascade_props_degree.pkl.gz"%(self.output_dir))
        return cascade_props_degree
    
    def get_cascade_inf_branching(self):
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

        cascade_props_degree1=self.cascade_props.query('inf_rootID==True').groupby("level")["degree"].apply(list).reset_index(name="degreeV")
        cascade_props_degree1=cascade_props_degree1.apply(_get_prob_vector,axis=1)
        cascade_props_degree1.set_index('level',inplace=True)
        cascade_props_degree1.to_pickle("%s/cascade_props_inf_degree.pkl.gz"%(self.output_dir))
        
        cascade_props_degree2=self.cascade_props.query('inf_rootID==False').groupby("level")["degree"].apply(list).reset_index(name="degreeV")
        cascade_props_degree2=cascade_props_degree2.apply(_get_prob_vector,axis=1)
        cascade_props_degree2.set_index('level',inplace=True)
        cascade_props_degree2.to_pickle("%s/cascade_props_non_inf_degree.pkl.gz"%(self.output_dir))
        return cascade_props_degree1,cascade_props_degree2
    
    def get_cascade_delays(self):
        cascade_props_size=self.cascade_props.groupby("rootID").size().reset_index(name="size")
        cascade_props_delay=self.cascade_props.groupby("rootID")["long_delay"].apply(list).reset_index(name="delayV")
        cascade_props_delay=pd.merge(cascade_props_delay,cascade_props_size,on="rootID",how="inner")
        cascade_props_delay.to_pickle("%s/cascade_props_delay.pkl.gz"%(self.output_dir))
        return cascade_props_delay


#cpath='/data/DevHub/recur-cascades/WWW2021/cpec_resources/cascade_records_0105_0106.pkl.gz'
cpath='/data/DevHub/recur-cascades/WWW2021/cpec_resources/cascade_records_0106_0107.pkl.gz'
#cpath='/data/DevHub/recur-cascades/WWW2021/cpec_resources/cascade_records_XhYAfMuJfDEBxtk_YuRV4A.pkl.gz'

cas=Cascade("cpec","twitter-gt",cpath)
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
print("saved, cascade branching")
cascade_props_inf_degree,cascade_props_non_inf_degree=cas.get_cascade_inf_branching()
print("saved, cascade inf branching")
cascade_props_delay=cas.get_cascade_delays()
print("saved, cascade delays")