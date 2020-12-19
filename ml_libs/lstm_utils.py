import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
Prepare input features for Training LSTM
'''        
def getFeatureTargetSequenceLSTM(x,xnarrative,target,features_local=[], features_global=[], time_in=0, time_out=0, narrative_list=[]):
    '''
    Description: Generates a sequence of n_input days features and target variable at n_out days for a particular frame.
    
    Input:
        x: timestamp
        xnarrative: frame to be processed
        target: target dataframe
        features_local: list of features per frame
        features_global: list of global features
        narrative_list: list of frames (all frames must start with 'informationID_')
        time_in: window for previous day features
        time_out: window for multi-step prediction
    
    '''
    ### target variable
    try:
        ### get target vector.
        target_variable=target.loc[x:x+timedelta(days=time_out-1)][xnarrative].values
    except:
        target_variable=[0]*time_out
    
    ### create one-hot encoding vector
    Tindex=narrative_list.index(xnarrative)
    narrative_binary=np.zeros(len(narrative_list))
    narrative_binary[Tindex]=1
    narrative_binary=np.tile(narrative_binary,(time_in,1))
    
    features=0
    
    ### append all local features corresponding to time-lag sequence
    for i,df in enumerate(features_local):
        if i == 0:
            features=df.loc[x-timedelta(days=time_in):x-timedelta(days=1)][[xnarrative]].values
        else:
            features_=df.loc[x-timedelta(days=time_in):x-timedelta(days=1)][[xnarrative]].values
            features=np.concatenate([features,features_], axis=1)
        
    ### append all global features corresponding to time-lag
    for i, df in enumerate(features_global):
        if type(features)==int and i==0:
            features=df.loc[x-timedelta(days=time_in):x-timedelta(days=1)].values
        else:
            features_=df.loc[x-timedelta(days=time_in):x-timedelta(days=1)].values
            if features_.ndim<2:
                features_=features_.reshape(len(features_),1)     
            features=np.concatenate([features,features_], axis=1)
        
    ### append one-hot encoding vector
    features=np.concatenate([features,narrative_binary],axis=1)

    return features,target_variable

'''
Prepare input features for Simulation LSTM
'''
def getSimulationSequenceLSTM(x,xnarrative,target,features_local=[], features_global=[], time_in=0, time_out=0, narrative_list=[]):
    '''
    Description: Used for simulation purposes. It generates a sequence of n_input days features and target variable at n_out days for a particular frame.
    
    Input:
        x: timestamp
        xnarrative: frame to be processed
        target: target dataframe
        features_local: list of features per frame
        features_global: list of global features
        narrative_list: list of frames (all frames must start with 'informationID_')
        time_in: window for previous day features
        time_out: window for multi-step prediction
    
    '''
    ### target variable
    try:
        ### get target vector.
        target_variable=target.loc[x:x+timedelta(days=time_out-1)][xnarrative].values
    except:
        ### if time is out of training range
        target_variable=[0]*time_out
    
    ### create one-hot encoding vector
    Tindex=narrative_list.index(xnarrative)
    narrative_binary=np.zeros(len(narrative_list))
    narrative_binary[Tindex]=1
    narrative_binary=np.tile(narrative_binary,(time_in,1))
    
    features=0
    
    ### append all local features corresponding to time-lag sequence
    for i,df in enumerate(features_local):
        if i == 0:
            try:
                ### get features within time range
                features=df.loc[x-timedelta(days=time_in):x-timedelta(days=1)][[xnarrative]].values
            except:
                try:
                    ### get features if exist within range and pad with np.NaN
                    features=df.loc[x-timedelta(days=time_in):][[xnarrative]].values
                    pad = time_in - features.shape[0]
                    features=np.expand_dims(np.append(features, np.array([np.NaN]*pad)), axis=1)
                except:  
                    ### pad with np.NaNs
                    features=np.array([[np.NaN]]*time_in)
        else:
            try:
                ### get features within time range
                features_=df.loc[x-timedelta(days=time_in):x-timedelta(days=1)][[xnarrative]].values
            except:
                try:
                    ### get features if exist within range and pad with np.NaN as placeholder
                    features_=df.loc[x-timedelta(days=time_in):][[xnarrative]].values
                    pad = time_in - features_.shape[0]
                    features_=np.expand_dims(np.append(features_, np.array([np.NaN]*pad)), axis=1)
                except:    
                    features_=np.array([[np.NaN]]*time_in) 
            features=np.concatenate([features,features_], axis=1)
        
    ### append all global features corresponding to time-lag
    for i, df in enumerate(features_global):
        if type(features)==int and i==0:
            features=df.loc[x-timedelta(days=time_in):x-timedelta(days=1)].values
        else:
            features_=df.loc[x-timedelta(days=time_in):x-timedelta(days=1)].values
            features=np.concatenate([features,features_], axis=1)

        
    ### append one-hot encoding vector
    features=np.concatenate([features,narrative_binary],axis=1)

    return features,target_variable


'''
Data preparation for LSTM
'''
def data_prepare_LSTM(sim_start_date, sim_end_date, target, features_local=[],
                    features_global=[], time_in=0, time_out=0, narrative_list=[]):
    '''
    Description: split data into input features and target for all frames
    
    Input:
        sim_start_date: starting time
        sim_end_date: ending time
        target: target dataframe
        features_local: list of features per frame
        features_global: list of global features
        narrative_list: list of frames (all frames must start with 'informationID_')
        time_in: window for previous day features
        time_out: window for multi-step prediction
    
    Output:
    data_X, data_y
    '''
    
    number_features_to_scale = len(features_local)+len(features_global)

    narrative_data={}
    for Tnarrative in narrative_list:
        data_X=[]
        data_y=[]

        print(Tnarrative)
        narrative_data_={}
        narrative_data.setdefault(Tnarrative,narrative_data_)
        for x in datespan(sim_start_date,sim_end_date-timedelta(days=time_out-2),delta=timedelta(days=1)):
            features,target_variable=getFeatureTargetSequenceLSTM(x,Tnarrative, target, narrative_list=narrative_list,
                                                              features_local=features_local, features_global=features_global,
                                                              time_in=time_in, time_out=time_out)

            data_X.append(features)
            data_y.append(target_variable)

        data_X=np.array(data_X)
        data_y=np.array(data_y)
        
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
        #data_y=np.append(data_y,data_y_)
        data_y.extend(data_y_)
    data_X=np.array(data_X)
    data_y=np.array(data_y)
    print(data_X.shape,data_y.shape)
    return data_X, data_y

'''
Get Replay Baseline predictions
'''
def getReplayBaselinePredictions(window_start_date, window_end_date,target, narrative_list=[]):
    sim_days = window_end_date - window_start_date
    sim_days = sim_days.days + 1

    ## Replay.
    baseline_window_end_date=(window_start_date-timedelta(days=1)).strftime("%Y-%m-%d")
    baseline_window_start_date=(window_start_date-timedelta(days=sim_days)).strftime("%Y-%m-%d")
    baseline_target=target[baseline_window_start_date:baseline_window_end_date]
    
    narrative_baseline_data={}
    for Tnarrative in narrative_list:
        y_hat=baseline_target[Tnarrative].values
        narrative_baseline_data.setdefault(Tnarrative,y_hat)
    return narrative_baseline_data

'''
Get Replay Baseline predictions
'''
def getSamplingBaselinePredictions(window_start_date, window_end_date,target, narrative_list=[]):
    sim_days = window_end_date - window_start_date
    sim_days = sim_days.days + 1

    ## Daily rate.
    baseline_window_end_date=(window_start_date-timedelta(days=1)).strftime("%Y-%m-%d")
    ##baseline_window_start_date=(window_start_date-timedelta(days=sim_days+1)).strftime("%Y-%m-%d")
    baseline_target=target[:baseline_window_end_date]
    baseline_target=baseline_target.groupby(baseline_target.index.weekday_name).mean()
    
    narrative_baseline_data={}
    next_weekday_names=[]
    for x in datespan(window_start_date,window_end_date+timedelta(days=1),delta=timedelta(days=1)):
        next_weekday_names.append(x.strftime("%A"))
    ##next_weekday_names=pd.DataFrame(next_weekday_names,columns=['dow'])
        
    for Tnarrative in narrative_list:
        y_hat=baseline_target[Tnarrative].loc[next_weekday_names].values
        narrative_baseline_data.setdefault(Tnarrative,y_hat)
    return narrative_baseline_data

'''
Evaluate baseline Predictions
'''
def eval_predictions(model_id,gt_data, sim_data, narrative_list=[]):
    '''
    Description: Run predictions for LSTM models
    
    Input:
    model_id: 
    gt_data: ground truth count dict
    sim_data: simulation count dict
    target: target dataframe
    narrative_list: list of info ids
    '''
    Gperformance_data=[]

    for Tnarrative in narrative_list:
        sim_y=gt_data[Tnarrative]
        y_hat=sim_data[Tnarrative]
        
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
    
    return Gperformance_data

'''
Run LSTM Predictions
'''
def run_predictions_LSTM(model_id, model, window_start_date, window_end_date, target, narrative_list=[], features_local=[],
                    features_global=[], time_in=0, time_out=0):
    '''
    Description: Run predictions for LSTM models
    
    Input:
    model_id: 
    model: the actual trained model
    window_start_date: starting date of simulation
    window_end_date: ending date of simulation
    target: target dataframe
    narrative_list: list of info ids
    features_local: list of local (per narrative) features
    features_global: list of global (per narrative) features
    time_in: window for previous day features
    time_out: window for multi-step prediction
    '''

    
    Gperformance_data=[]
    narrative_sim_data={}
    narrative_gt_data={}
    
    number_features_to_scale = len(features_local)+len(features_global)
    for Tnarrative in narrative_list:
        sim_X=[]
        sim_y=[]
        ### Skip time based on the n_out window
        for x in datespan(window_start_date,window_end_date+timedelta(days=1),delta=timedelta(days=time_out)):

            features,target_variable=getSimulationSequenceLSTM(x,Tnarrative, target, narrative_list=narrative_list,
                                                          features_local=features_local, features_global=features_global,
                                                          time_in=time_in, time_out=time_out)
            sim_X.append(features)
            sim_y.append(target_variable)
        sim_X=np.array(sim_X)
        sim_y=np.array(sim_y)
        sim_y=sim_y.reshape(-1)

        
        sim_X[:,:,:number_features_to_scale] = np.log1p(sim_X[:,:,:number_features_to_scale])

        #sim_X[np.isnan(sim_X)] = 0
        print(sim_X.shape,len(sim_y))

        y_hat=[]
        y_hat_norm=[]
        for i in range(sim_X.shape[0]):
            
            sim_X_ = sim_X[i]
            missing = len(sim_X_[np.isnan(sim_X_)])
            if missing > 0:
                sim_X_[np.isnan(sim_X_)] = y_hat_norm[-missing:]
            
            yhat=model.predict(np.expand_dims(sim_X_,axis=0))

            y_hat.extend(np.round(np.expm1(yhat[0])))
            y_hat_norm.extend(np.log1p(np.round(np.expm1(yhat[0]))))

        narrative_sim_data.setdefault(Tnarrative,y_hat)
        narrative_gt_data.setdefault(Tnarrative,sim_y) 

    
    return narrative_sim_data, narrative_gt_data