import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

'''
Metrics
'''

def normed_rmse(v1,v2):
    v1=np.cumsum(v1)
    v2=np.cumsum(v2)
    v1=v1/np.max(v1)
    v2=v2/np.max(v2)
    
    result = v1-v2
    result = (result ** 2).mean()
    result = np.sqrt(result)
    return result

def rmse(v1,v2):
    result = np.array(v1)-np.array(v2)
    result = (result ** 2).mean()
    result = np.sqrt(result)
    return result

def ape(v1,v2):
    v1=np.sum(v1)
    v2=np.sum(v2)
    result = np.abs(float(v1) - float(v2))
    result = 100.0 * result / np.abs(float(v1))
    return result

def smape(A, F):
    A=np.array(A)
    F=np.array(F)
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

'''
Time lag functions
'''
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def datespan(startDate, endDate, delta=timedelta(days=7)):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta
        
'''
Preprocess features and target variables
'''

def getFeatureTargetDay(x,xnarrative,target,features_local=[], features_global=[], delta=(1,0),narrative_list=[]):
    '''
    Description: Generates previous day features and target variable for a particular frame.
    
    Input:
        x: timestamp
        xnarrative: frame to be processed
        target: target dataframe
        features_local: list of features per frame
        features_global: list of global features
        delta: indicates the time lag in (days, hours)
        narrative_list: list of frames (all frames must start with 'informationID_')
    
    '''
    ### target variable
    target_variable=target.loc[x][xnarrative]
    
    ### create one-hot encoding vector
    Tindex=narrative_list.index(xnarrative)
    narrative_binary=np.zeros(len(narrative_list))
    narrative_binary[Tindex]=1
    
    features=0
    
    ### append all local features corresponding to time-lag
    for i,df in enumerate(features_local):
        if i == 0:
            features=df.loc[x-timedelta(days=delta[0], hours=delta[1])][[xnarrative]].values
        else:
            features_=df.loc[x-timedelta(days=delta[0], hours=delta[1])][[xnarrative]].values
            features=np.append(features,features_)
        
    ### append all global features corresponding to time-lag
    for i, df in enumerate(features_global):
        if type(features)==int and i==0:
            features=df.loc[x-timedelta(days=delta[0], hours=delta[1])].values
        else:
            features_=df.loc[x-timedelta(days=delta[0], hours=delta[1])].values
            features=np.append(features,features_)
        
    ### append one-hot encoding vector
    features=np.append(features,narrative_binary)

    return features,target_variable

def getFeatureTargetSequence(x,xnarrative,target,features_local=[], features_global=[], delta=(1,0),narrative_list=[]):
    '''
    Description: Generates a sequence of previous day features and target variable at next day for a particular frame.
    
    Input:
        x: timestamp
        xnarrative: frame to be processed
        target: target dataframe
        features_local: list of features per frame
        features_global: list of global features
        delta: indicates the time lag in (days, hours)
        narrative_list: list of frames (all frames must start with 'informationID_')
    
    '''
    ### target variable
    target_variable=target.loc[x][xnarrative]
    
    ### create one-hot encoding vector
    Tindex=narrative_list.index(xnarrative)
    narrative_binary=np.zeros(len(narrative_list))
    narrative_binary[Tindex]=1
    narrative_binary=np.tile(narrative_binary,(delta[0],1))
    
    features=0
    
    ### append all local features corresponding to time-lag sequence
    for i,df in enumerate(features_local):
        if i == 0:
            features=df.loc[x-timedelta(days=delta[0], hours=delta[1]):x-timedelta(days=1)][[xnarrative]].values
        else:
            features_=df.loc[x-timedelta(days=delta[0], hours=delta[1]):x-timedelta(days=1)][[xnarrative]].values
            features=np.concatenate([features,features_], axis=1)
        
    ### append all global features corresponding to time-lag
    for i, df in enumerate(features_global):
        if type(features)==int and i==0:
            features=df.loc[x-timedelta(days=delta[0], hours=delta[1]):x-timedelta(days=1)].values
        else:
            features_=df.loc[x-timedelta(days=delta[0], hours=delta[1]):x-timedelta(days=1)].values
            features=np.concatenate([features,features_], axis=1)
        
    ### append one-hot encoding vector
    features=np.concatenate([features,narrative_binary],axis=1)

    return features,target_variable

def data_split(sim_start_date, sim_end_date, target, features_local=[],
                    features_global=[], delta=(1,0), narrative_list=[]):
    '''
    Description: split data into input features and target for all frames
    
    Input:
        sim_start_date: starting time
        sim_end_date: ending time
        target: target dataframe
        features_local: list of features per frame
        features_global: list of global features
        delta: indicates the time lag in (days, hours)
        narrative_list: list of frames (all frames must start with 'informationID_')
        
    '''
    
    number_features_to_scale = len(features_local)+len(features_global)

    narrative_data={}
    for Tnarrative in narrative_list:
        data_X=[]
        data_y=[]

        print(Tnarrative)
        narrative_data_={}
        narrative_data.setdefault(Tnarrative,narrative_data_)
        for x in datespan(sim_start_date,sim_end_date,delta=timedelta(days=1)):
            if delta[0]==1:
                features,target_variable=getFeatureTargetDay(x,Tnarrative, target, narrative_list=narrative_list,
                                                              features_local=features_local, features_global=features_global,
                                                              delta=delta)
            else:
                features,target_variable=getFeatureTargetSequence(x,Tnarrative, target, narrative_list=narrative_list,
                                                              features_local=features_local, features_global=features_global,
                                                              delta=delta)

            data_X.append(features)
            data_y.append(target_variable)

        data_X=np.array(data_X)
        data_y=np.array(data_y)
        
        if delta[0] == 1:

            ### log-normalization - skip one-hot encoder
            data_X[:,:number_features_to_scale] = np.log1p(data_X[:,:number_features_to_scale])
            data_y = np.log1p(data_y)
        else: #sequences rather than one day
            data_X[:,:,:number_features_to_scale] = np.log1p(data_X[:,:,:number_features_to_scale])
            data_y = np.log1p(data_y)


        data_X[np.isnan(data_X)] = 0
        data_y[np.isnan(data_y)] = 0
        print(data_X.shape,data_y.shape)

        narrative_data_['data_X']=data_X
        narrative_data_['data_y']=data_y
        
    data_X=[]
    data_y=[]
    for T1narrative in narrative_list:
        print("Train on narrative: %s"%T1narrative)
        data_X_=narrative_data[T1narrative]['data_X']
        data_X.extend(data_X_)
        data_y_=narrative_data[T1narrative]['data_y']
        data_y=np.append(data_y,data_y_)
    data_X=np.array(data_X)
    print(data_X.shape,data_y.shape)
    return data_X, data_y

def run_predictions(model_id, model, window_start_date, window_end_date, target, narrative_list=[], features_local=[],
                    features_global=[], delta=(1,0)):
    '''
    Input:
    model_id: 
    model: the actual trained model
    window_start_date: starting date of simulation
    window_end_date: ending date of simulation
    target: target dataframe
    narrative_list: list of info ids
    features_local: list of local (per narrative) features
    features_global: list of global (per narrative) features
    delta: time lag
    '''

    
    Gperformance_data=[]
    narrative_sim_data={}
    narrative_gt_data={}
    
    number_features_to_scale = len(features_local)+len(features_global)
    for Tnarrative in narrative_list:
        sim_X=[]
        sim_y=[]
        for x in datespan(window_start_date,window_end_date+timedelta(days=1),
                          delta=timedelta(days=1)):
            if delta[0]==1:
                features,target_variable=getFeatureTargetDay(x,Tnarrative, target, narrative_list=narrative_list,
                                                              features_local=features_local, features_global=features_global,
                                                              delta=delta)
            else:
                features,target_variable=getFeatureTargetSequence(x,Tnarrative, target, narrative_list=narrative_list,
                                                              features_local=features_local, features_global=features_global,
                                                              delta=delta)
            sim_X.append(features)
            sim_y.append(target_variable)
        sim_X=np.array(sim_X)

        if delta[0] == 1:

            ### log-normalization - skip one-hot encoder
            sim_X[:,:number_features_to_scale] = np.log1p(sim_X[:,:number_features_to_scale])

        else: #sequences rather than one day
            sim_X[:,:,:number_features_to_scale] = np.log1p(sim_X[:,:,:number_features_to_scale])

        sim_X[np.isnan(sim_X)] = 0
        print(sim_X.shape,len(sim_y))

        y_hat=[]
        for i in range(sim_X.shape[0]):
            yhat=model.predict(np.expand_dims(sim_X[i],axis=0))

            y_hat.append(np.round(np.expm1(yhat[0])))

        narrative_sim_data.setdefault(Tnarrative,y_hat)
        narrative_gt_data.setdefault(Tnarrative,sim_y) 
        
        print("Test on narrative: %s"%Tnarrative)
        ape_value=ape(sim_y,y_hat)
        print("APE: ",ape_value)

        rmse_value=rmse(sim_y,y_hat)
        print("RMSE: ",rmse_value)

        nrmse_value=normed_rmse(sim_y,y_hat)
        print("NRMSE: ",nrmse_value)

        smape_value=smape(sim_y,y_hat)
        print("SMAPE: ",smape_value)

        Gperformance_data.append([Tnarrative,ape_value,rmse_value,nrmse_value,smape_value,model_id])

    Gperformance_data=pd.DataFrame(Gperformance_data,columns=['informationID','APE','RMSE','NRMSE','SMAPE','MODEL'])
    
    return Gperformance_data, narrative_sim_data, narrative_gt_data

def run_simulations(model_id, model, window_start_date, window_end_date, target, narrative_list=[], features_local=[],
                    features_global=[], delta=(1,0)):

    narrative_sim_data={}

    
    number_features_to_scale = len(features_local)+len(features_global)
    for Tnarrative in narrative_list:
        sim_X=[]
        sim_y=[]
        for x in datespan(window_start_date,window_end_date+timedelta(days=1),
                          delta=timedelta(days=1)):
            if delta[0]==1:
                features,target_variable=getFeatureTargetDay(x,Tnarrative, target, narrative_list=narrative_list,
                                                              features_local=features_local, features_global=features_global,
                                                              delta=delta)
            else:
                features,target_variable=getFeatureTargetSequence(x,Tnarrative, target, narrative_list=narrative_list,
                                                              features_local=features_local, features_global=features_global,
                                                              delta=delta)
            sim_X.append(features)
            sim_y.append(target_variable)
        sim_X=np.array(sim_X)

        if delta[0] == 1:

            ### log-normalization - skip one-hot encoder
            sim_X[:,:number_features_to_scale] = np.log1p(sim_X[:,:number_features_to_scale])

        else: #sequences rather than one day
            sim_X[:,:,:number_features_to_scale] = np.log1p(sim_X[:,:,:number_features_to_scale])

        sim_X[np.isnan(sim_X)] = 0
        print(sim_X.shape,len(sim_y))

        y_hat=[]
        for i in range(sim_X.shape[0]):
            yhat=model.predict(np.expand_dims(sim_X[i],axis=0))

            y_hat.append(np.round(np.expm1(yhat[0])))

        narrative_sim_data.setdefault(Tnarrative,y_hat) 
     
    
    return narrative_sim_data