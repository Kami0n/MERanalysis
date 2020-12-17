import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

clear = lambda: os.system('cls')
clear()
print("\n"*5)

import numpy as np
import pandas as pd
import keras
import sklearn.preprocessing as skPre

from pickle import load
from os import listdir
from os.path import isfile, join
from commonFunctions import musicFeatureExtraction
from commonFunctions import displayVAgraph

#spremenljivke
from commonFunctions import subfolderName
from commonFunctions import showResults

subfolder = subfolderName

onlyfiles = [f for f in listdir(subfolder) if isfile(join(subfolder, f)) and ".mp3" in f]
#onlyfiles = ['Frank Sinatra - New York New York.mp3','Ritchie Blackmore - Snowman.mp3']

allFeatures = []
for fileName in onlyfiles:
    print(fileName)
    data = musicFeatureExtraction(subfolder+str(fileName))
    allFeatures.append([fileName, data])

featuresdf = pd.DataFrame(allFeatures, columns=['id','features'])
featuresdf = featuresdf.set_index(['id'])
#print(featuresdf)

X_pred = np.array(featuresdf['features'].tolist()) # input
#print(X_pred)
#scaler = load(open('model_NN_valence_arousal_normalized_scaler.pkl', 'rb'))
#X_pred_scaled = scaler.transform(X_pred)
#print(X_pred_scaled)

model_valence = keras.models.load_model('model_NN_10fold_valence')
model_arousal = keras.models.load_model('model_NN_10fold_arousal')

predictions_valence = model_valence.predict(X_pred)
predictions_arousal = model_arousal.predict(X_pred)



# od tukaj do printa je samo za prikaz !
npValence = np.round(np.array(predictions_valence), 2).T
npArousal = np.round(np.array(predictions_arousal),2).T
combined = np.vstack((npValence, npArousal)).T

names = [x.replace('.mp3', '') for x in onlyfiles]

dataValenceArousal = pd.DataFrame(data=combined, index=names, columns=['valence', 'arousal']) 
print(dataValenceArousal.sort_index())

# displayVAgraph(valence, arousal, names, min, max)
displayVAgraph( predictions_valence, predictions_arousal, names, -1, 1 )

if showResults:
    from commonFunctions import rezultatiTestData
    rezultatiTestData(predictions_valence, predictions_arousal)