import pandas as pd
import numpy as np

import random
import networkx as nx
import math

import time, math
import json
import glob
import os
import pickle
from datetime import datetime, timedelta, date
from collections import Counter
import networkx as nx

"""Helper Functions"""
def convert_datetime(dataset, verbose):
    """
    Description:
    Input:
    Output:
    """

    if verbose:
        print('Converting strings to datetime objects...', end='', flush=True)


    try:
        dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'], unit='s')
    except:
        try:
            dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'], unit='ms')
        except:
            dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'])
        
    dataset['nodeTime'] = dataset['nodeTime'].dt.tz_localize(None)

    dataset['nodeTime'] = dataset['nodeTime'].dt.tz_localize(None)

    if verbose:
        print(' Done')
        
    return dataset

def create_dir(x_dir):
    if not os.path.exists(x_dir):
        os.makedirs(x_dir)
        print("Created new dir. %s"%x_dir)
    else:
        print("Dir. already exists")
        
def get_parent_uids(df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", user_col="nodeUserID"):
    """
    :return: adds parentUserID column with user id of the parent if it exits in df_
    if it doesn't exist, uses the user id of the root instead
    if both doesn't exist: NaN
    """
    df_ = df.copy()
    
    tweet_uids = pd.Series(df_[user_col].values, index=df_[node_col]).to_dict()

    df_['parentUserID'] = df_[parent_node_col].map(tweet_uids)

    df_.loc[(df_[root_node_col] != df_[node_col]) & (df_['parentUserID'].isnull()), 'parentUserID'] = df_[(df_[root_node_col] != df_[node_col]) & (df_['parentUserID'].isnull())][root_node_col].map(tweet_uids)

    df_ = df_[df_['nodeUserID'] != df_['parentUserID']]

    return df_

def get_random_id():
    hash = random.getrandbits(64)
    return "%16x"%hash

def get_random_ids(size):
    return [get_random_id() for i in range(size)]

def get_random_id_new_user():
    hash = random.getrandbits(64)
    return "new_%16x"%hash

def get_random_new_user_ids(size):
    return [get_random_id_new_user() for i in range(size)]

def getActProbSetOldUsers(df_, users):
    """
    df: dataframe of simulation outputs for a particular infoid 
    users: list of users to get probability from
    
    return:
    user_dict: list of old users keyed per topic
    prob_dict: list of probabilities based on act. level keyed per topic
    """
    
    df = df_.copy()
    
    ### Remove new users in cascade ouputs from probability
    df = df.loc[df["nodeUserID"].isin(users)].reset_index(drop=True)
    ### Count the number of total activities per topic
    total_acts = df.groupby("informationID")["nodeID"].nunique().reset_index(name="total_count")
    ### Count old user activities per topic
    df_act = df.groupby(["informationID", "nodeUserID"])["nodeID"].nunique().reset_index(name="count")
    
    df_act = pd.merge(df_act, total_acts, on="informationID", how="left")
    df_act["prob"] = df_act["count"]/df_act["total_count"]
    
    df_act = df_act.sort_values(["informationID", "prob"], ascending=[True, False]).reset_index(drop=True)
    
    user_dict = df_act.groupby("informationID")["nodeUserID"].apply(list)
    user_dict = dict(zip(user_dict.index, user_dict.values))
    prob_dict = df_act.groupby("informationID")["prob"].apply(list)
    prob_dict = dict(zip(prob_dict.index, prob_dict.values))
    
    return user_dict, prob_dict

"""End of Helper Functions"""

"""User Replacement Helper Functions"""
def newuser_replacement(df_, df_nusers_, infoid, platform="", seedType="", responseType="", tmp_path="", conflict_path=""):
    
    """
    df_: cascade output file
    df_nusers_: dataframe for predicted values
    """
    conflict_dict = {}
    
    ### Global variables to hold df outputs
    concat_to_finaldf = []
    
    df = df_.copy()
    df_nusers = df_nusers_.copy()
    n = infoid
    
    ### Get simulation periods
    periods = sorted(list(set(df_nusers["nodeTime"])))
    ### Get records for platform
    df = df.query("platform==@platform").reset_index(drop=True)
    ### Get new user predictions for platform and informationID
    df_nusers = df_nusers.query("platform==@platform and informationID==@n").reset_index(drop=True)
    df_nusers = df_nusers.set_index("nodeTime")
    
    ### Extract seeds records for informationID
    df_seeds = df.query("actionType==@seedType and informationID==@n").reset_index(drop=True)
    ### Extract response records for informationID
    df_responses = df.query("actionType==@responseType and informationID==@n").reset_index(drop=True)
    
    ### Iterate in a timely-based manner
    for period in periods:
        
        ### Obtain predicted number of new users at particular period
        num_nu = int(df_nusers.loc[period]["new_users"])
        
        ### Obtain records pertain to new users already in cascade responses outputs
        df_nusers_cas = df_responses.query("nodeTime==@period").reset_index(drop=True)
        df_nusers_cas = df_nusers_cas.loc[df_nusers_cas["nodeUserID"].str.match(r'^new_')==True].reset_index(drop=True)
        ### Get list of new users in cascade responses
        list_nu_cas = list(df_nusers_cas["nodeUserID"].unique())
        ### Get number of new users in cascade responses
        num_nu_cas = int(df_nusers_cas["nodeUserID"].nunique())
        
        ### Obtain records in cascade responses without new users (i.e., only old records)
        df_ousers_cas = df_responses.query("nodeTime==@period").reset_index(drop=True)
        df_ousers_cas = df_ousers_cas.loc[~df_ousers_cas["nodeUserID"].isin(list_nu_cas)].reset_index(drop=True)
        
        ### Get difference between predicted new users and new users already in system
        diff_nusers = num_nu - num_nu_cas
        
        if diff_nusers == 0: ### There are no conflicts 
            print("InfoID: {0}, Predicted new users and new users already in the system are equal...".format(n))
            ### Append again all records in response cascades
            concat_to_finaldf.append(df_ousers_cas)
            concat_to_finaldf.append(df_nusers_cas)
        elif diff_nusers < 0: ### There are more new users in the system than predicted
            print("InfoID: {0}, Predicted new users is LESS than new users already in the system...".format(n))
            key = "Prediction < Cur New Users"
            conflict_dict.setdefault(key, 0)
            conflict_dict[key] += 1
            
            ### Number of users we need to replace with old identity
            num_replace = int(abs(diff_nusers))
            
            ### Rank new users cascade outputs by cascade size on a particular time
            rank_cas = df_nusers_cas.groupby(["parentUserID", "parentID", "rootUserID", "rootID"])["nodeID"].nunique().reset_index(name="cascade_size")
            df_nusers_cas = pd.merge(df_nusers_cas, rank_cas, on=["parentUserID", "parentID", "rootUserID", "rootID"], how="left")
            
            ### Replace new users with old user identities from smaller cascades first
            df_nusers_cas = df_nusers_cas.sort_values("cascade_size", ascending=True).reset_index(drop=True)
            df_nusers_cas.loc[0:num_replace-1, "nodeUserID"] = "old_" + df_nusers_cas["nodeUserID"]
            
            ### Drop cascade size attribute
            df_nusers_cas = df_nusers_cas.drop(columns=["cascade_size"])
            ### Append again all records in response cascades
            concat_to_finaldf.append(df_nusers_cas)
            concat_to_finaldf.append(df_ousers_cas)
        else: ### There are more new users predicted than there are in the system, so we need to add more
            print("InfoID: {0}, Predicted new users is GREATER than new users already in the system...".format(n))
            key = "Prediction < Cur New Users"
            conflict_dict.setdefault(key, 0)
            conflict_dict[key] += 1
            
            num_add = int(abs(diff_nusers))
            
            ### Retrieve most recent cascades
            recent_cas = df_responses.query("nodeTime<=@period").reset_index(drop=True)
            
            ### If there are previous records, we can attach new users to previous cascades
            if len(recent_cas)!=0: ### No records at all from cascade outputs. but there are previous records.
                print("InfoID: {0}, Cascade Outputs did not predict any records...".format(n))
                key = "Only Prev Cascade Output Records"
                conflict_dict.setdefault(key, 0)
                conflict_dict[key] += 1
                
                ### Add completely new records to most recent and larger cascade
                # rank cascades and retrieve parent and root information 
                rank_cas = recent_cas.groupby(["parentUserID", "parentID", "rootUserID", "rootID"])["nodeID"].nunique().reset_index(name="cascade_size")
                recent_cas = pd.merge(recent_cas, rank_cas, on=["parentUserID", "parentID", "rootUserID", "rootID"], how="left")
                recent_cas = recent_cas.sort_values(["nodeTime","cascade_size"], ascending=[False,False]).reset_index(drop=True)
                recent_cas = recent_cas.loc[0:0]
                if num_add > 1:
                    recent_cas = recent_cas.append([recent_cas]*(num_add-1),ignore_index=True)
                nodeuserids = get_random_new_user_ids(num_add)
                new_nodeids = get_random_ids(num_add)
                ### Change proper columns
                recent_cas["nodeUserID"] = nodeuserids
                recent_cas["actionType"] = responseType
                recent_cas["nodeID"] = new_nodeids
                recent_cas["nodeTime"] = period
                recent_cas = recent_cas.drop(columns=["cascade_size"])
                concat_to_finaldf.append(recent_cas)
                concat_to_finaldf.append(df_nusers_cas)
                concat_to_finaldf.append(df_ousers_cas)
                
            else: ### There are no previous records, so we need to add completely new records
                print("InfoID: {0},{1}, Cascade Outputs have no records at all...".format(n, period))
                key = "No Cascade Output Records"
                conflict_dict.setdefault(key, 0)
                conflict_dict[key] += 1
                dict_records = dict()
                new_nodeids = get_random_ids(num_add)
                new_rootids = new_nodeids
                new_parentids = new_nodeids
                nodeuserids = get_random_new_user_ids(num_add)
                rootuserids = nodeuserids
                parentuserids = nodeuserids
                dict_records['rootID'] = new_rootids
                dict_records['parentID'] = new_parentids
                dict_records['nodeID'] = new_nodeids
                dict_records['nodeUserID'] = nodeuserids
                dict_records['parentUserID']=parentuserids
                dict_records['rootUserID'] = rootuserids
                dict_records['informationID'] = [n]*num_add
                dict_records['platform'] = [platform]*num_add
                dict_records['nodeTime'] = [period]*num_add
                dict_records['actionType'] = [responseType]*num_add
                sample_records = pd.DataFrame(dict_records, columns=columns)
                concat_to_finaldf.append(sample_records)
                concat_to_finaldf.append(df_nusers_cas)
                concat_to_finaldf.append(df_ousers_cas)
    
    final_df = pd.concat(concat_to_finaldf, ignore_index=True, sort=True)
    final_df = pd.concat([df_seeds, final_df], ignore_index=True, sort=True)
    final_df = final_df.sort_values('nodeTime').reset_index(drop=True)
    filename = n.replace('/', '-')
    final_df.to_pickle(tmp_path+'_'+filename+'_'+platform+'.pkl.gz')
    
    if conflict_path != "":
        with open(conflict_path+filename+'_newuser.pkl.gz', "wb") as f:
            pickle.dump(conflict_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    del final_df
    del conflict_dict
    
def olduser_replacement(df_, df_ousers_, infoid, platform="", seedType="", responseType="", tmp_path="", conflict_path=""):
    """
    df_: simulation cascade output file
    df_ousers_: dataframe with old users predictions
    """
    conflict_dict = {}
    
    df = df_.copy()
    df_ousers = df_ousers_.copy()
    n = infoid
    
    ### Global variables to hold df outputs
    concat_to_finaldf = []
    
    ### Get simulation periods
    periods = sorted(list(set(df_ousers["nodeTime"])))
    ### Get records for platform
    df = df.query("platform==@platform and informationID==@n").reset_index(drop=True)
    ### Get new user predictions for platform and informationID
    df_ousers = df_ousers.query("platform==@platform and informationID==@n").reset_index(drop=True)
    df_ousers = df_ousers.set_index("nodeTime")
    
    for period in periods:
        
        ### Extract all new users records from df in particular period
        df_nusers_cas = df.query("nodeTime==@period").reset_index(drop=True)
        df_nusers_cas = df_nusers_cas.loc[df_nusers_cas["nodeUserID"].str.match(r'^new_')==True].reset_index(drop=True)

        ### Extract all old users records from df in particular period
        df_ousers_cas = df.query("nodeTime==@period").reset_index(drop=True)
        df_ousers_cas = df_ousers_cas.loc[df_ousers_cas["nodeUserID"].str.match(r'^new_')==False].reset_index(drop=True)
        
        ### Extract seeds from old users 
        df_ou_seeds = df_ousers_cas.query("actionType==@seedType").reset_index(drop=True)
        ### Extract responses from old users
        df_ou_responses = df_ousers_cas.query("actionType==@responseType").reset_index(drop=True)
        
        ### Obtain predicted number of new users at particular period
        num_ou = int(df_ousers.loc[period]["old_users"])
        
        ### Check number of old users in seeds already in system
        num_ou_seeds = int(df_ou_seeds["nodeUserID"].nunique())
        ### Check number of old users in responses
        num_ou_reponses = int(df_ou_responses["nodeUserID"].nunique())
        
        
        ### Difference between predicted old users and old users in seeds
        diff_n = num_ou - num_ou_seeds
        
        if diff_n == 0: ### There are enough oldies in seeds already
            print("InfoID: {0},{1}, There are enough old users in seeds...".format(n, period))
            k = "Old users in seed equal to predictions"
            conflict_dict.setdefault(k, 0)
            conflict_dict[k] +=1
            
            ### Replace all oldies in responses with seed users based on activity probability
            list_ou_seeds = list(df_ou_seeds["nodeUserID"].unique())
            user_dict, prob_dict = getActProbSetOldUsers(df, list_ou_seeds)
            users_list = user_dict[n]
            prob_list = prob_dict[n]
            
            ### Get list of users we need to replace
            list_ou_responses = list(df_ou_responses["nodeUserID"].unique())
            ### Draw with replacement
            new_user_ids = random.choices(users_list, weights=prob_list, k=len(list_ou_responses))
            ### Map old users in responses to new old identities
            new_user_ids_map = dict(zip(list_ou_responses, new_user_ids))
            df_ou_responses["nodeUserID"] = df_ou_responses["nodeUserID"].map(new_user_ids_map).fillna(df_ou_responses['nodeUserID'])
            df_nusers_cas["parentUserID"] = df_nusers_cas["parentUserID"].map(new_user_ids_map).fillna(df_nusers_cas["parentUserID"])
            df_nusers_cas["rootUserID"] = df_nusers_cas["rootUserID"].map(new_user_ids_map).fillna(df_nusers_cas["rootUserID"])
            concat_to_finaldf.append(df_nusers_cas)
            concat_to_finaldf.append(df_ou_seeds)
            concat_to_finaldf.append(df_ou_responses)
        elif diff_n < 0: ### There are too many oldies in seeds, we need to trim and replace all user identities for responses
            print("InfoID: {0},{1}, Need to TRIM old users in seeds...".format(n, period))
            k = "Old users in seed > than predictions"
            conflict_dict.setdefault(k, 0)
            conflict_dict[k] +=1
            ### Replace those users with low cascade activity
            list_ou_seeds = list(df_ou_seeds["nodeUserID"].unique())
            rank_seed_users = df.query("nodeTime==@period").reset_index(drop=True)
            ### Obtain cascade size for each seed user
            rank_seed_users =rank_seed_users.loc[rank_seed_users["rootUserID"].isin(list_ou_seeds)].reset_index(drop=True)
            rank_seed_users=rank_seed_users.groupby("rootUserID")["nodeID"].nunique().reset_index(name="size")
            rank_seed_users = rank_seed_users.sort_values("size", ascending=False).reset_index(drop=True)
            ### Pick the users in tail to replace
            replace_users = list(rank_seed_users.tail(int(abs(diff_n)))["rootUserID"])
            keep_users = list(rank_seed_users.loc[~rank_seed_users["rootUserID"].isin(replace_users)]["rootUserID"])
            
            ### Get activity probability of these users to keep                     
            user_dict, prob_dict = getActProbSetOldUsers(df, keep_users)
            users_list = user_dict[n]
            prob_list = prob_dict[n]
            ### Get new ids for users to replace
            new_seeds_ids = np.random.choice(users_list, size=int(abs(diff_n)), replace=True, p=prob_list)
            ### Get mapping
            new_user_ids_map = dict(zip(replace_users, new_seeds_ids))
            
            df_ou_seeds["nodeUserID"] = df_ou_seeds["nodeUserID"].map(new_user_ids_map).fillna(df_ou_seeds['nodeUserID'])
            df_ou_seeds["parentUserID"] = df_ou_seeds["parentUserID"].map(new_user_ids_map).fillna(df_ou_seeds['parentUserID'])
            df_ou_seeds["rootUserID"] = df_ou_seeds["rootUserID"].map(new_user_ids_map).fillna(df_ou_seeds['rootUserID'])
            
            ### Change responses identities
            ### Get list of users we need to replace
            list_ou_responses = list(df_ou_responses["nodeUserID"].unique())
            ### Draw with replacement
            new_user_ids = random.choices(users_list, weights=prob_list, k=len(list_ou_responses))
            ### Map old users in responses to new old identities
            new_user_ids_resp_map = dict(zip(list_ou_responses, new_user_ids))
            df_ou_responses["nodeUserID"] = df_ou_responses["nodeUserID"].map(new_user_ids_resp_map).fillna(df_ou_responses['nodeUserID'])
            df_ou_responses["parentUserID"] = df_ou_responses["parentUserID"].map(new_user_ids_map).fillna(df_ou_responses['parentUserID'])
            df_ou_responses["rootUserID"] = df_ou_responses["rootUserID"].map(new_user_ids_map).fillna(df_ou_responses['rootUserID'])
            df_nusers_cas["parentUserID"] = df_nusers_cas["parentUserID"].map(new_user_ids_map).fillna(df_nusers_cas["parentUserID"])
            df_nusers_cas["rootUserID"] = df_nusers_cas["rootUserID"].map(new_user_ids_map).fillna(df_nusers_cas["rootUserID"])
            
            concat_to_finaldf.append(df_nusers_cas)
            concat_to_finaldf.append(df_ou_responses)
            concat_to_finaldf.append(df_ou_seeds)
        else: ### There are more old users predicted than there are old seed users
            print("InfoID: {0},{1}, There are more old users predicted than seeds...".format(n, period))
            k = "Old users in seed < than predictions"
            conflict_dict.setdefault(k, 0)
            conflict_dict[k] +=1
            ### Check difference between old users predicted vs. old users in responses
            diff_m = diff_n - num_ou_reponses
            
            if diff_m == 0: ### Old users in cascade outputs and predictions match
                print("InfoID: {0},{1}, There are enough old users in responses...".format(n, period))
                k = "Old users in responses equal to predictions"
                conflict_dict.setdefault(k, 0)
                conflict_dict[k] +=1
                ### Only need to remove those users with old_ tag in responses if any
                old_tag_users_df = df_ou_responses.loc[df_ou_responses["nodeUserID"].str.match(r'^old_')==True].reset_index(drop=True)
                if len(old_tag_users_df) > 0:
                    old_tag_users = list(old_tag_users_df["nodeUserID"].unique())
                    ### Get activity probability of all old users not already in outputs
                    users = set(df.loc[(df["nodeUserID"].str.match(r'^old_')==False)&(df["nodeUserID"].str.match(r'^new_')==False)]["nodeUserID"].unique())
                    users = users - set(df_ousers_cas["nodeUserID"].unique())
                    user_dict, prob_dict = getActProbSetOldUsers(df, users)
                    users_list = user_dict[n]
                    prob_list = prob_dict[n]
                    ### Get old users without replacement
                    new_user_ids = np.random.choice(users_list, size=len(old_tag_users), replace=False, p=prob_list)
                    new_user_ids_map = dict(zip(old_tag_users, new_user_ids))
                    df_ou_responses["nodeUserID"] = df_ou_responses["nodeUserID"].map(new_user_ids_map).fillna(df_ou_responses["nodeUserID"])
                concat_to_finaldf.append(df_nusers_cas)
                concat_to_finaldf.append(df_ou_seeds)
                concat_to_finaldf.append(df_ou_responses)
            elif diff_m < 0: ### We need to reduce responses since there are more old users than predicted
                print("InfoID: {0},{1}, There are more old users in system than predicted in responses...".format(n, period))
                k = "Old users in responses > than predictions"
                conflict_dict.setdefault(k, 0)
                conflict_dict[k] +=1
                ### Replace all old users in responses with new old user identities
                old_users_to_replace = list(df_ou_responses["nodeUserID"].unique())
                ### Take out users in seeds from pool
                users = set(df.loc[(df["nodeUserID"].str.match(r'^old_')==False)&(df["nodeUserID"].str.match(r'^new_')==False)]["nodeUserID"].unique())
                users = users - set(df_ou_seeds["nodeUserID"].unique())
                user_dict, prob_dict = getActProbSetOldUsers(df, users)
                users_list = user_dict[n]
                prob_list = prob_dict[n]
                            
                ### Get old users without replacement (only the amount of users we need from predictions)
                new_user_ids = np.random.choice(users_list, size=int(diff_n), replace=False, p=prob_list)
                ### Now assign to all response user one of this old users
                new_user_ids = np.random.choice(new_user_ids, size=len(old_users_to_replace), replace=True)
                            
                new_user_ids_map = dict(zip(old_users_to_replace, new_user_ids))
                df_ou_responses["nodeUserID"] = df_ou_responses["nodeUserID"].map(new_user_ids_map).fillna(df_ou_responses["nodeUserID"])
                
                concat_to_finaldf.append(df_nusers_cas)
                concat_to_finaldf.append(df_ou_seeds)
                concat_to_finaldf.append(df_ou_responses)
            else: ### There are more old users predicted than in current responses
                print("InfoID: {0},{1}, There are more old users predicted than in system...".format(n, period))
                k = "Old users in responses < than predictions"
                conflict_dict.setdefault(k, 0)
                conflict_dict[k] +=1
                ### First make sure to assign old_user tags with old_users identities if any
                old_tag_users_df = df_ou_responses.loc[df_ou_responses["nodeUserID"].str.match(r'^old_')==True].reset_index(drop=True)
                old_tag_users = list(old_tag_users_df["nodeUserID"].unique())
                ### Get activity probability of all old users not already in outputs
                users = set(df.loc[(df["nodeUserID"].str.match(r'^old_')==False)&(df["nodeUserID"].str.match(r'^new_')==False)]["nodeUserID"].unique())
                users = users - set(df_ousers_cas["nodeUserID"].unique())
                if len(old_tag_users_df) > 0:
                    user_dict, prob_dict = getActProbSetOldUsers(df, users)
                    users_list = user_dict[n]
                    prob_list = prob_dict[n]
                    ### Get old users without replacement
                    new_user_ids = np.random.choice(users_list, size=len(old_tag_users), replace=False, p=prob_list)
                    new_user_ids_map = dict(zip(old_tag_users, new_user_ids))
                    df_ou_responses["nodeUserID"] = df_ou_responses["nodeUserID"].map(new_user_ids_map).fillna(df_ou_responses["nodeUserID"])
                    
                ### Now introduce the remaining old users needed to most recent largest cascade (ignore those newly introduced old users)
                users = users - set(df_ou_responses["nodeUserID"])  
                user_dict, prob_dict = getActProbSetOldUsers(df, users)
                users_list = user_dict[n]
                prob_list = prob_dict[n]
                
                new_old_user_ids = np.random.choice(users_list, size=int(diff_m), replace=False, p=prob_list)
                
                ### Retrieve most recent cascades
                recent_cas = df.query("nodeTime<=@period").reset_index(drop=True)
                if len(recent_cas) != 0: ### Attatch old users to largest cascade
                    rank_cas = recent_cas.groupby(["parentUserID", "parentID", "rootUserID", "rootID"])["nodeID"].nunique().reset_index(name="cascade_size")
                    recent_cas = pd.merge(recent_cas, rank_cas, on=["parentUserID", "parentID", "rootUserID", "rootID"], how="left")
                    recent_cas = recent_cas.sort_values(["nodeTime","cascade_size"], ascending=[False,False]).reset_index(drop=True)
                    recent_cas = recent_cas.loc[0:0]
                    if diff_m > 1:
                        recent_cas = recent_cas.append([recent_cas]*(diff_m-1),ignore_index=True)
                    new_nodeids = get_random_ids(diff_m)
                    ### Change proper columns
                    recent_cas["nodeUserID"] = new_old_user_ids
                    recent_cas["actionType"] = responseType
                    recent_cas["nodeID"] = new_nodeids
                    recent_cas["nodeTime"] = period
                    recent_cas = recent_cas.drop(columns=["cascade_size"])
                    concat_to_finaldf.append(recent_cas)
                    concat_to_finaldf.append(df_ou_responses)
                    concat_to_finaldf.append(df_nusers_cas)
                    concat_to_finaldf.append(df_ou_seeds)
                else: ### Introduce completely new records
                    dict_records = dict()
                    new_nodeids = get_random_ids(diff_m)
                    new_rootids = new_nodeids
                    new_parentids = new_nodeids
                    nodeuserids = new_old_user_ids
                    rootuserids = nodeuserids
                    parentuserids = nodeuserids
                    dict_records['rootID'] = new_rootids
                    dict_records['parentID'] = new_parentids
                    dict_records['nodeID'] = new_nodeids
                    dict_records['nodeUserID'] = nodeuserids
                    dict_records['parentUserID']=parentuserids
                    dict_records['rootUserID'] = rootuserids
                    dict_records['informationID'] = [n]*diff_m
                    dict_records['platform'] = [platform]*diff_m
                    dict_records['nodeTime'] = [period]*diff_m
                    dict_records['actionType'] = [responseType]*diff_m
                    sample_records = pd.DataFrame(dict_records, columns=columns)
                    concat_to_finaldf.append(sample_records)
                    concat_to_finaldf.append(df_nusers_cas)
                    concat_to_finaldf.append(df_ou_responses)
                    concat_to_finaldf.append(df_ou_seeds)
    
    final_df = pd.concat(concat_to_finaldf, ignore_index=True, sort=True)
    final_df = final_df.sort_values('nodeTime').reset_index(drop=True)
    filename = n.replace('/', '-')
    final_df.to_pickle(tmp_path+'_'+filename+'_'+platform+'.pkl.gz')
    
    if conflict_path != "":
        with open(conflict_path+filename+'_olduser.pkl.gz', "wb") as f:
            pickle.dump(conflict_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    del final_df
    del conflict_dict
"""End of user replacement helper functions"""

def newuser_replacement_v2(df_, df_nusers_, infoid, platform="", seedType="", responseType="", tmp_path="", conflict_path=""):
    
    """
    df_: cascade output file
    df_nusers_: dataframe for predicted values
    """
    conflict_dict = {}
    
    ### Global variables to hold df outputs
    concat_to_finaldf = []
    
    df = df_.copy()
    df_nusers = df_nusers_.copy()
    n = infoid
    
    ### Get simulation periods
    periods = sorted(list(set(df_nusers["nodeTime"])))
    ### Get records for platform
    df = df.query("platform==@platform and informationID==@n").reset_index(drop=True)
    ### Get new user predictions for platform and informationID
    df_nusers = df_nusers.query("platform==@platform and informationID==@n").reset_index(drop=True)
    df_nusers = df_nusers.set_index("nodeTime")
    
    ### Extract seeds records for informationID
    df_seeds = df.query("actionType==@seedType").reset_index(drop=True)
    ### Extract response records for informationID
    df_responses = df.query("actionType==@responseType").reset_index(drop=True)
    
    ### Iterate in a timely-based manner
    for period in periods:
        
        ### Get seeds at this period
        df_seeds_cas = df_seeds.query("nodeTime==@period").reset_index(drop=True)
        ### Get responses at this period
        df_responses_cas = df_responses.query("nodeTime==@period").reset_index(drop=True)
        
        ### Obtain predicted number of new users at particular period
        num_nu = int(df_nusers.loc[period]["new_users"])
        
        ### No new users predicted
        if num_nu == 0:
            print("InfoID: {0},{1}, No new users predicted...".format(n, period))
            concat_to_finaldf.append(df_seeds_cas)
            concat_to_finaldf.append(df_responses_cas)
            continue
         
         ### Conflict if number of new users predicted exceeds responses on this day
        if num_nu > len(df_responses_cas):
            print("InfoID: {0},{1}, New Users predicted is greater than number of records...".format(n, period))
            ### 1. replace all users in responses with new identities
            ### 2. Introduce new users proportionally to high in-degree users in previous cascades, if any
            ### 3. Introduce completely new users and records
            new_userids = get_random_new_user_ids(int(len(df_responses_cas)))
            df_responses_cas["nodeUserID"] = new_userids
            ### Difference
            diff_nu = num_nu - len(df_responses_cas)
            ### Introduce new users
            df_prev = df_responses.query("nodeTime<=@period").reset_index(drop=True)
            df_add = addNewUsers(df_prev, diff_nu, n, period,responseType, platform)
            ### Concat to final df
            concat_to_finaldf.append(df_seeds_cas)
            concat_to_finaldf.append(df_responses_cas)
            concat_to_finaldf.append(df_add)
        ### Conflict, there are enough responses to introduce new users by replacing old users
        elif num_nu <= len(df_responses_cas):
            print("InfoID: {0},{1}, New Users predicted is less than number of records...".format(n, period))
            n_users = int(df["nodeUserID"].nunique())
            in_deg_parent_df = df_responses_cas.groupby("parentUserID")["nodeUserID"].nunique().reset_index(name="in_deg")
            in_deg_parent_df = dict(zip(in_deg_parent_df["parentUserID"], in_deg_parent_df["in_deg"]))
#             user_out_degree_df = df_responses_cas.groupby("nodeUserID")["nodeUserID"].nunique().reset_index(name="out_deg")
#             user_out_degree_df = dict(zip(user_out_degree_df["nodeUserID"], user_out_degree_df["out_deg"]))
            df_responses_cas["in_deg"] = df_responses_cas["parentUserID"].map(in_deg_parent_df)
#             df_responses_cas["out_deg"] = df_responses_cas["nodeUserID"].map(user_out_degree_df)
            ### Sort by parentUser in-degree and then replace old with new users
            df_responses_cas=df_responses_cas.sort_values("in_deg", ascending=False).reset_index(drop=True)
            new_userids = get_random_new_user_ids(num_nu)
            df_responses_cas.loc[0:num_nu-1, "nodeUserID"] = new_userids
            df_responses_cas=df_responses_cas.drop(columns=["in_deg"])
            ### concat to final df
            concat_to_finaldf.append(df_seeds_cas)
            concat_to_finaldf.append(df_responses_cas)
    final_df = pd.concat(concat_to_finaldf, ignore_index=True, sort=True)
#     final_df = pd.concat([df_seeds, final_df], ignore_index=True, sort=True)
    final_df = final_df.sort_values('nodeTime').reset_index(drop=True)
    filename = n.replace('/', '-')
    final_df.to_pickle(tmp_path+'_'+filename+'_'+platform+'.pkl.gz')
    
    if conflict_path != "":
        with open(conflict_path+filename+'_newuser.pkl.gz', "wb") as f:
            pickle.dump(conflict_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    del final_df
    del conflict_dict
    
def addNewUsers(df_, k, n, period, responseType, platform):
    """
    df: dataframe with records up to period currently processing
    k: number of new users to introduce
    n: narrative
    period: nodeTime
    responseType: actionType
    platform
    
    return: df with new users added
    """
    df = df_.copy()
    ### Check length of df is not equal 0 proceed, else return completely new identities
    ### Construct in-degree and weights by dividing in-degree by total number of new users
    ### Draw a list of parentUserIDs and retrieve for each parent a random parentID
    if len(df)!=0:
        print("InfoID: {0},{1}, Adding New Users to largest cascades...".format(n, period))
        
        in_deg_df = df.groupby("parentUserID")["nodeUserID"].nunique().reset_index(name="in_deg")
        n_users = in_deg_df["in_deg"].sum()
        in_deg_df["weight"] = in_deg_df["in_deg"]/n_users
        users = list(in_deg_df["parentUserID"])
        prob_list = list(in_deg_df["weight"])
        ### Draw with replacement k parentUsers
        parents = np.random.choice(users, size=int(k), replace=True, p=prob_list)
        parents_count = Counter(parents)
        ### Draw sample records in df
        records = []
        for parent, count in parents_count.items():
            new_nodeids = get_random_ids(count)
            nodeuserids = get_random_new_user_ids(count)
            record = df.query("parentUserID==@parent").sample(n=count, replace=True).reset_index(drop=True)
            record["nodeUserID"] = nodeuserids
            record["nodeID"] = new_nodeids
            record["nodeTime"] = period
            record["actionType"] = responseType
            records.append(record)
        final_df = pd.concat(records,ignore_index=True)
    else: ### there are no previous cascades add completely new users
        print("InfoID: {0},{1}, No previous cascades: Adding Completely New Users Records...".format(n, period))
        dict_records = dict()
        new_nodeids = get_random_ids(k)
        new_rootids = new_nodeids
        new_parentids = new_nodeids
        nodeuserids = get_random_new_user_ids(k)
        rootuserids = nodeuserids
        parentuserids = nodeuserids
        dict_records['rootID'] = new_rootids
        dict_records['parentID'] = new_parentids
        dict_records['nodeID'] = new_nodeids
        dict_records['nodeUserID'] = nodeuserids
        dict_records['parentUserID']=parentuserids
        dict_records['rootUserID'] = rootuserids
        dict_records['informationID'] = [n]*k
        dict_records['platform'] = [platform]*k
        dict_records['nodeTime'] = [period]*k
        dict_records['actionType'] = [responseType]*k
        final_df = pd.DataFrame(dict_records, columns=columns)
    return final_df

def newuser_replacement_v3(df_, df_nusers_, deg_dict, infoid, platform="", seedType="", responseType="", tmp_path="", conflict_path=""):
    
    """
    df_: cascade output file
    df_nusers_: dataframe for predicted values
    """
    conflict_dict = {}
    
    ### Global variables to hold df outputs
    concat_to_finaldf = []
    
    df = df_.copy()
    df_nusers = df_nusers_.copy()
    n = infoid
    
    ### Get simulation periods
    periods = sorted(list(set(df_nusers["nodeTime"])))
    ### Get records for platform
    df = df.query("platform==@platform and informationID==@n").reset_index(drop=True)
    ### Get new user predictions for platform and informationID
    df_nusers = df_nusers.query("platform==@platform and informationID==@n").reset_index(drop=True)
    df_nusers = df_nusers.set_index("nodeTime")
    
    if (platform=="youtube") and (n=="chinese-affiliated-account"):
        concat_to_finaldf.append(df)
    else:
        ### Iterate in a timely-based manner
        for period in periods:

            ### Obtain predicted number of new users at particular period
            num_nu = int(df_nusers.loc[period]["new_users"])

            ### Get daily records
            df_cas = df.query("nodeTime==@period").reset_index(drop=True)
            ### Shuffle dataframe
            df_cas = df_cas.sample(frac=1).reset_index(drop=True)
            ### Obtain user degrees
            topic_dict = deg_dict[n]
            df_cas["out_deg"] = df_cas["nodeUserID"].map(topic_dict)
            ### Sort values by out degree remove those with large out degrees
            df_cas = df_cas.sort_values("out_deg", ascending=True).reset_index(drop=True)

            new_userids = get_random_new_user_ids(num_nu)

            ### No new users predicted
            if num_nu == 0:
                print("InfoID: {0},{1}, No new users predicted...".format(n, period))
                df_cas= df_cas.drop(columns=["out_deg"])
                concat_to_finaldf.append(df_cas)
                continue

            ### No conflicts 
            if num_nu <= len(df_cas):
                print("InfoID: {0},{1}, New Users predicted is less than number of records...".format(n, period))
                ### Replace old users with new user identities based on degrees
                df_cas.loc[0:num_nu-1, "nodeUserID"] = new_userids
                df_cas= df_cas.drop(columns=["out_deg"])
                concat_to_finaldf.append(df_cas)
            elif num_nu > len(df_cas): ### New users predicted is greater than total shares
                print("InfoID: {0},{1}, New Users predicted is greater than number of records...".format(n, period))

                new_userids = get_random_new_user_ids(int(len(df_cas)))
                df_cas["nodeUserID"] = new_userids
                ### Difference
                diff_nu = num_nu - len(df_cas)
                ### Introduce new users
                df_prev = df.query("nodeTime<=@period").reset_index(drop=True)
                df_add = addNewUsers(df_prev, diff_nu, n, period,responseType, platform)
                ### Concat to final df
                df_cas= df_cas.drop(columns=["out_deg"])
                concat_to_finaldf.append(df_cas)
                concat_to_finaldf.append(df_add)
    final_df = pd.concat(concat_to_finaldf, ignore_index=True, sort=True)
    final_df = final_df.sort_values('nodeTime').reset_index(drop=True)
    filename = n.replace('/', '-')
    final_df.to_pickle(tmp_path+'_'+filename+'_'+platform+'.pkl.gz')
    
    if conflict_path != "":
        with open(conflict_path+filename+'_newuser.pkl.gz', "wb") as f:
            pickle.dump(conflict_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    del final_df
    del conflict_dict
    
def getInDegrees(df):
    """
    df: dataframe for whole period of simulation
    return: dict keyed by infoids, and another dict keyed by nodeUserID and degree.
    """
    infoids = df["informationID"].unique()
    deg_dict = {}
    for infoid in infoids:
        deg_dict.setdefault(infoid, {})
        tmp = df.query("informationID==@infoid").reset_index(drop=True)
        edgelist_df = tmp.groupby(['nodeUserID','parentUserID']).size().reset_index(name='weight')
        G = nx.from_pandas_edgelist(edgelist_df, 'nodeUserID', 'parentUserID', ['weight'], create_using=nx.DiGraph())
        tmp_dict = {}
        for (node, val) in G.in_degree(): 
            tmp_dict[node] = val
        deg_dict[infoid] = tmp_dict
        
    return deg_dict

def getOutDegrees(df):
    """
    df: dataframe for whole period of simulation
    return: dict keyed by infoids, and another dict keyed by nodeUserID and degree.
    """
    infoids = df["informationID"].unique()
    deg_dict = {}
    for infoid in infoids:
        deg_dict.setdefault(infoid, {})
        tmp = df.query("informationID==@infoid").reset_index(drop=True)
        edgelist_df = tmp.groupby(['nodeUserID','parentUserID']).size().reset_index(name='weight')
        G = nx.from_pandas_edgelist(edgelist_df, 'nodeUserID', 'parentUserID', ['weight'], create_using=nx.DiGraph())
        tmp_dict = {}
        for (node, val) in G.out_degree(): 
            tmp_dict[node] = val
        deg_dict[infoid] = tmp_dict
        
    return deg_dict