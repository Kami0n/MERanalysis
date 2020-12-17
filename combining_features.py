import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
import librosa
import csv

from commonFunctions import musicFeatureExtraction
from commonFunctions import displayVAgraph

# function to parse CSV file of valence and arousal
def parseDEAM(pathEmotions):
    emotions = pd.read_csv(pathEmotions, index_col=0, sep=',')
    return emotions

subfolder = 'Dataset/'


emotions = parseDEAM(subfolder+'emotions/filtered_annotations_vse.csv') # filtered_annotations_vse  valence_arousal_vse_normalized
print(emotions['valence_mean'])

featuresdf = pd.read_pickle(subfolder+'pickle/exported_features_valence_arousal_normalized.pkl')


mypath = subfolder+'metadata/'
with open(mypath+'allGeneres.csv', newline='', encoding='utf-8') as csvfile:
    fileReader = pd.read_csv(csvfile, sep=',')
    ids = fileReader.values[:,0]
    allGeneres = fileReader.values[:,1]

myset = set(allGeneres)
print(myset)

numberGenere = []
for genere in allGeneres:
    numberGenere.append()

featuresdf_new = featuresdf.drop(columns=['valence', 'arousal'])

featuresdf_new['genere'] = allGeneres
featuresdf_new['valence'] = emotions['valence_mean'].tolist()
featuresdf_new['arousal'] = emotions['arousal_mean'].tolist()
print(featuresdf_new)

featuresdf_new.to_pickle(subfolder+'pickle/features_genere_valence_arousal.pkl')