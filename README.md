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
call the 'preprocessing' function. It remove needless features.(median_house_value) and fill the missing values


## Clustering
We defined functions to cluster automatically with computing all combination of parameters that specified scaler, models and hyperparameters. It performs clustering and plotting with various models and hyperparameter values.



## Compare the clustering results 
Compare the results with N quantiles of the medianHouseValue feature values in the original dataset. In this case, we compared with N=4, N=5,and N=8.


