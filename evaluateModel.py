import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print("\n"*5)

import numpy as np
import pandas as pd
import keras
import sklearn.preprocessing as skPre
import joblib

from pickle import load
from os import listdir
from os.path import isfile, join
from commonFunctions import musicFeatureExtraction
from commonFunctions import displayVAgraph

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

np.set_printoptions(suppress=True)

def normalizacija(array, faktor):
    return 2.*(array - np.min(array))/np.ptp(array)-faktor


subfolder = 'Dataset/'
featuresdf = pd.read_pickle(subfolder+'pickle/features_genere_valence_arousal.pkl')

X = np.array(featuresdf['features'].tolist())
Y_valence = np.array(featuresdf['valence'].tolist())
Y_arousal = np.array(featuresdf['arousal'].tolist())

Y_valence = normalizacija(Y_valence, 1)
Y_arousal = normalizacija(Y_arousal, 1)

model_valence = joblib.load("./RFmodel/modelSVMValence.joblib")
model_arousal = joblib.load("./RFmodel/modelSVMArousal.joblib")


y = Y_valence
seed = 0
kf = KFold(n_splits=10, shuffle = True, random_state = seed)
best = None
i = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    predictions = model_valence.predict(X_test)
    error = round(mean_squared_error(y_test, predictions),4)
    
    if(best == None or best > error):
        best = error
    print(i, 'Mean Squared Error:', error)
    i+=1
print('Best MSE:', best)