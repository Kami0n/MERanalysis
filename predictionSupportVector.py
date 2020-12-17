import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n"*100)
clear = lambda: os.system('cls')
clear()

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

#spremenljivke
from commonFunctions import subfolderName
from commonFunctions import showResults

subfolder = subfolderName

onlyfiles = [f for f in listdir(subfolder) if isfile(join(subfolder, f)) and ".mp3" in f]
#onlyfiles = ['Ritchie Blackmore - Snowman.mp3']#'Frank Sinatra - New York New York.mp3',

allFeatures = []
for fileName in onlyfiles:
    print(fileName)
    data = musicFeatureExtraction(subfolder+str(fileName))
    allFeatures.append([fileName, data])

featuresdf = pd.DataFrame(allFeatures, columns=['id','features'])
featuresdf = featuresdf.set_index(['id'])
#print(featuresdf)

X_pred = np.array(featuresdf['features'].tolist()) # input

model_valence = joblib.load("./RFmodel/modelSVMValence.joblib")
model_arousal = joblib.load("./RFmodel/modelSVMArousal.joblib")

predictions_valence = model_valence.predict(X_pred)
predictions_arousal = model_arousal.predict(X_pred)

# od tukaj do printa je samo za prikaz !
npValence = np.round(np.array(predictions_valence), 2)
npArousal = np.round(np.array(predictions_arousal),2)
combined = np.vstack((npValence, npArousal)).T

names = [x.replace('.mp3', '') for x in onlyfiles]

dataValenceArousal = pd.DataFrame(data=combined, index=names, columns=['valence', 'arousal']) 
print(dataValenceArousal.sort_index())

# displayVAgraph(valence, arousal, names, min, max)
#displayVAgraph( predictions_valence, predictions_arousal, names, -1, 1 )

if showResults:
    from commonFunctions import rezultatiTestData
    rezultatiTestData(predictions_valence, predictions_arousal)