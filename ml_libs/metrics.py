import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dtw

from datetime import datetime, timedelta, date

'''
Metrics
'''

def myerrsq(x,y):
    return((x-y)**2)

### s2 predictions, s1 ground truth
def dtw_(s1, s2):
    window=2
    
    s1= pd.DataFrame(s1)
    s2 = pd.DataFrame(s2)
    
    z1=(s1-s1.mean())/(s1.std(ddof=0).apply(lambda m: (m if m > 0.0 else 1.0)))
    z2=(s2-s2.mean())/(s2.std(ddof=0).apply(lambda m: (m if m > 0.0 else 1.0)))

    ### first value simulation second value GT
    dtw_metric = np.sqrt(dtw.dtw(z2[0], z1[0], dist_method=myerrsq, window_type='slantedband',
                               window_args={'window_size':window}).normalizedDistance)
    
    return dtw_metric

def ae(v1,v2):
    v1=np.array(v1)
    v2 = np.array(v2)
    return np.abs(v1 - v2)

# Scale-Free Absolute Error
def sfae(v1,v2):
    
    v1=np.array(v1)
    v2 = np.array(v2)
    
    return ae(v1, v2) / np.mean(v1)

def MAD_mean_ratio(v1, v2):
    """
    MAD/mean ratio
    """
    return np.mean(sfae(v1, v2))

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