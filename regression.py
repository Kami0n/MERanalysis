import os
clear = lambda: os.system('cls')
clear()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from commonFunctions import displayVAgraph

subfolder = 'Dataset/'

featuresdf = pd.read_pickle(subfolder+'pickle/exported_features_valence_arousal_normalized.pkl')
#print(featuresdf)

X = np.array(featuresdf['features'].tolist()) # input
Y = np.array([featuresdf['valence'].tolist(), featuresdf['arousal'].tolist()]) # output
y = Y.T


train_d, test_d, train_l, test_l = train_test_split(X, y, test_size=0.2, random_state=0)

result = []
xlabel = [i for i in range(1, 11)]
for neighbors in range(1, 11):
    kNN = KNeighborsClassifier(n_neighbors=neighbors)
    kNN.fit(train_d, train_l)
    prediction = kNN.predict(test_d)
    result.append(accuracy_score(prediction, test_l)*100)

plt.figure(figsize=(10, 10))
plt.xlabel('kNN Neighbors for k=1,2...20')
plt.ylabel('Accuracy Score')
plt.title('kNN Classifier Results')
plt.ylim(0, 100)
plt.xlim(0, xlabel[len(xlabel)-1]+1)
plt.plot(xlabel, result)
plt.savefig('1-fold 10NN Result.png')
plt.show()