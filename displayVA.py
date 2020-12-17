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
from commonFunctions import normalizacija
from commonFunctions import parseDEAM



emotions = parseDEAM('dataset2/izbrane_15.csv')

Y_valence = np.array(emotions['valence_mean'].tolist())
Y_arousal = np.array(emotions['arousal_mean'].tolist())

Y_valence = normalizacija(Y_valence, 1, 0, 1)
Y_arousal = normalizacija(Y_arousal, 1, 0, 1)

# y=mx
m=1

# Define the domain of the function
xmin = -3.0
xmax = 3.0
step = 0.1

# This function uses a transformation matrix to return the point
# that is reflected across the line y=mx for m defined above.

def reflect(x,y):
    xhat = np.array([x,y]).T
    matrix = np.array([[1-m**2, 2*m],[2*m, m**2 -1]])
    matrix = matrix * (1/(1+m**2))
    
    R = matrix.dot(xhat)
    return R

# Here we use the transformation matrix function we defined above:
Z = zip(Y_valence,Y_arousal)

valence = []
arousal = []
for x,y in Z:
    r = reflect(x,y)
    valence.append(r[0])
    arousal.append(r[1])


displayVAgraph(valence, arousal, emotions.index.tolist(), -1, 1)


#df = pd.DataFrame({'id':emotions.index, 'valence':valence,'arousal':arousal})
#print(df)
#df.to_csv('out.csv', index=False)
