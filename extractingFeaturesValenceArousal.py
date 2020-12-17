import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import pyAudioAnalysis

from os import listdir
from os.path import isfile, join
from commonFunctions import musicFeatureExtraction
from commonFunctions import displayVAgraph

dirname = os.path.dirname(__file__)

# function to parse CSV file of valence and arousal
def parseDEAM(pathEmotions):
    #pathToEmotions = os.path.join(dirname,pathEmotions)
    emotions = pd.read_csv(pathEmotions, index_col=0, sep=',')
    return emotions

#subfolder = 'dataset_small/'
subfolder = 'Dataset/'

#emotions = parseDEAM(subfolder+'emotions/valence_arousal_vse_normalized.csv')

emotions = parseDEAM(subfolder+'emotions/filtered_annotations_vse.csv')
#print(emotions)

# displayVAgraph(valence, arousal, names, min, max)
#displayVAgraph(emotions['valence_mean'], emotions['arousal_mean'], False, 1, 9)

allFeatures = []

#filesTemp = [1000]
#for fileName in filesTemp:
for fileName, row in emotions.iterrows():
    print(fileName)
    fullFilePath = subfolder+'audio/'+str(fileName)+'.mp3'
    #print(fullFilePath)
    data = musicFeatureExtraction(fullFilePath)
    allFeatures.append([fileName, data, emotions.loc[fileName,'valence_mean'], emotions.loc[fileName,'arousal_mean']])

featuresdf = pd.DataFrame(allFeatures, columns=['id','features','valence', 'arousal'])
featuresdf = featuresdf.set_index(['id'])
print(featuresdf)

featuresdf.to_pickle(subfolder+'pickle/more_exported_features_valence_arousal.pkl')

