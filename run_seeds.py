import random
import pandas as pd
import numpy as np
from collections import Counter
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, date
import os,sys,pickle
from gensim.models import Word2Vec

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

sim_start_date="2019-02-15"
sim_end_date="2019-02-28"
sim_platform="twitter"
exog_taste="news"
exog_taste_="%s_pday_8pm"%exog_taste



sim_start_date=datetime.strptime(sim_start_date,"%Y-%m-%d")
sim_end_date=datetime.strptime(sim_end_date,"%Y-%m-%d")
oneD=timedelta(days=1)

sim_start_date_d_oned_text=(sim_start_date-oneD).strftime("%Y-%m-%d")

with open('../vz_prediction_seed_output/narrative_sim_data_seed_%s.pkl'%exog_taste_, 'rb') as f:
    narrative_seed_data=pickle.load(f) 
sim_narratives=list(narrative_seed_data.keys())

seed_users=pd.read_pickle("./metadata/probs/twitter-venezuela/influentials_followers.pkl.gz")
seed_users['normed_no_tweets']=seed_users['no_tweets']/seed_users['no_tweets'].sum()
#print(seed_users.head())        
# print("exog networks loading....",exog_taste)
# exog_network_dataset=pd.read_pickle("resources/%s_network_dataset.pkl.gz"%exog_taste)

# print("skipgrams loading....",exog_taste)
# skipgram_model=Word2Vec.load("resources/skipgram_%s_%s.model"%(exog_taste,sim_platform))
# ###### -------------------------------------------sim_start_date_d_oned_text







# def getUserIdentities(test_day_text):
#     ##print("Predicting seed user identities..")
#     test_day_text_start_epoch="%s 00:00"%test_day_text
#     test_day_text_end_epoch="%s 23:59"%test_day_text
#     exog_users=exog_network_dataset.loc[test_day_text_start_epoch:test_day_text_end_epoch].groupby('nodeUserID').size().sort_values(ascending=False)

#     exog_neighbors=[]
#     for guser,v in exog_users.iteritems():
#         if guser in skipgram_model.wv:
#             normal_neighbors = skipgram_model.wv.most_similar_cosmul([skipgram_model.wv[guser]], topn=500)
#             neighbors = pd.DataFrame(normal_neighbors, columns=['nodeUserID', 'hits'])
            
#             neighbors['hits']=neighbors['hits']*v
#             exog_neighbors.append(neighbors)

                
#     ## predicted
#     exog_neighbors=pd.concat(exog_neighbors).groupby('nodeUserID')['hits'].sum().reset_index()#.sort_values(by='hits',ascending=False)
    
#     ## Filtering, excluding exog users, and including only old seed users
#     #exog_neighbors=pd.merge(exog_neighbors,seed_users,left_index=True,right_index=True)
#     exog_neighbors=pd.merge(exog_neighbors,seed_users,on='nodeUserID',how='inner')
#     #exog_neighbors=exog_neighbors[exog_neighbors['nodeUserID'].isin(seed_users)==True]
        
#     exog_neighbors['hits']=exog_neighbors['normed_no_seeds']/exog_neighbors['normed_no_seeds'].sum()
#     ##exog_neighbors['hits']=exog_neighbors['hits']/exog_neighbors['hits'].sum()
#     ##exog_neighbors.reset_index(inplace=True)
#     print("# seed user identities: ",exog_neighbors.shape[0])
#     return exog_neighbors




filePath='output/seeds_%s_mcas_exog_taste_%s_dates_%s_%s.csv'%(sim_platform,exog_taste_,sim_start_date.strftime("%Y-%m-%d"),sim_end_date.strftime("%Y-%m-%d"))
if os.path.exists(filePath):
    os.remove(filePath)
    print("Existing file deleted.")
                                                               
fd=open(filePath,'a')

global_event_count=0
for Tnarrative in sim_narratives:
    index=0
    seed_data=narrative_seed_data[Tnarrative]
    ##seed_narrative_users=seed_users.loc[Tnarrative]
    for sim_day in daterange(sim_start_date, sim_end_date+oneD):
        sim_day_text=sim_day.strftime("%Y-%m-%d")
        seed_count=int(seed_data[index])
        print("Day: %s, Narrative: %s, # seeds: %d"%(sim_day_text,Tnarrative, seed_count))
        if seed_count<=0:
            continue
        global_event_count+=seed_count
        index+=1
        
        
#         ## who are the seed authors
#         seed_user_ids=getUserIdentities(sim_day_text)
#         ## oldies are half
#         no_oldies=int(seed_count/2)
#         seed_user_ids=list(seed_user_ids.sample(no_oldies, weights='hits', replace=True)['nodeUserID'])
#         ## newbies come
#         no_newbies=seed_count-no_oldies
#         seed_user_ids_=list(seed_users.sample(no_newbies, replace=True)['nodeUserID'])
#         ## oldies + newbies
#         seed_user_ids.extend(seed_user_ids_)

        seed_user_ids=list(seed_users.sample(seed_count, weights='normed_no_tweets',replace=True).index)

        print("%d authors perform %d seeds"%(len(set(seed_user_ids)),seed_count))
        assert(len(seed_user_ids)==seed_count)


        for seed_index in range(seed_count):
            seed_identifier = "seed_%16x"%random.getrandbits(64)
            seed_user_id=seed_user_ids[seed_index]
            fd.write("%s,%s,%s,%s\n"%(sim_day_text,seed_identifier,seed_user_id,Tnarrative))
            
            
print("Saved seeds at %s"%filePath)
print("Total # seeds: ",global_event_count)