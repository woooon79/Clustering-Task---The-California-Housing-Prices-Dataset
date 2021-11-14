# Clustering-Task : The-California-Housing-Prices-Dataset (PHW2)



## Before run this manual, please make sure the install and import following packages.
```
!pip install pyclustering
```
```
import random
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,RobustScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import *
from pyclustering.cluster.clarans import clarans;
from pyclustering.utils import timedcall;
from sklearn import datasets
from pyclustering.cluster import cluster_visualizer_multidim
```

## Loading a Dataset
```
df = pd.read_csv('/content/drive/MyDrive/housing.csv', delimiter = ",")
df_original = df.copy()

print(df.shape)
print(df.isnull().sum())
```


## Setting up the combinations.
You can freely add or delete the elements you want. 
```
encoders = [LabelEncoder(), OneHotEncoder()]
scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
models = ['K_Means','MeanShift','CLARANS','DBSCAN','GMM']
hyperparams = {
    #'K_Means_params':{}
    #'GMM_params':{}
    #'CLARANS_params':{}
    'DBSCAN_params': {
        'eps': [0.01, 0.003] 
    },
    'MeanShift_params': {
        'n': [10, 50, 100]
    },
    'k': range(2, 13)
}

```

## Clean and prepare a dataset 
Call the 'preprocessing' function. It remove needless features.(median_house_value) and fill the missing values
```
'''
<  preprocessing  >
 Input : df

- Remove needless features.(median_house_value)
- fill the missing values

 Output : modified dataframe
'''


def preprocessing(df):
    df.drop(columns=["median_house_value"], inplace=True)
    df.total_bedrooms.fillna(df.total_bedrooms.median(), inplace=True)

    return df

```

## Clustering
We defined functions to cluster automatically with computing all combination of parameters that specified scaler, models and hyperparameters. It performs clustering and plotting with various models and hyperparameter values.

> * df : dataset 
> * models : list of models
>   + ['K_Means',' MeanShift', 'CLARANS', 'DBSCAN', 'GMM'] 
> * hyperparams : list of models’ hyperparameters
>   + hyperparams = {
'DBSCAN_params': { 'eps': [0.01, 0.003] },
'MeanShift_params': { 'n': [10, 50, 100] },
'k': range(2, 13)
}




## Compare the clustering results 
Compare the results with N quantiles of the medianHouseValue feature values in the original dataset. In this case, we compared with N=4, N=5,and N=8.


## Execute
All of these processes are executed automatically by calling the main function.

```
'''
< main >

INPUT : df(dataframe), scalers(list), models(list), hyperparams(dict), combi(list)

- preprocessing
- scaling
- Call 'clustering()' function to cluster and plot

'''

def main(df, scalers, models, hyperparams, combi):
    new_df = preprocessing(df)
    for i in combi:
        X = new_df[i]

        print("Current combination", i)

        for scaler in scalers:
            print("Current scaler:", scaler)
            scaled_X = scaler.fit_transform(X)
            data_df = pd.DataFrame(scaled_X)
            clustering(data_df, models, hyperparams)
```
