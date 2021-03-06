import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
print("\n"*100)
clear = lambda: os.system('cls')
clear()

import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import sklearn.preprocessing as skPre
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from commonFunctions import displayVAgraph
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

from commonFunctions import normalizacija

from sklearn.model_selection import KFold

def showAccLoss(history):
    print(history.history.keys())
    #  "Accuracy"
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # "Loss"
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def showLoss(history):
    #print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def skaliranje(array):
    return 2.*(array - np.min(array))/np.ptp(array)-1

def trainModel(name, X, y, seed):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    
    #svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    #regr = make_pipeline(StandardScaler(), svr)
    
    regr = SVR(kernel='rbf', C=1.0, epsilon=0.1, verbose=True)
    
    regr.fit(X_train, y_train)
    
    predictions = regr.predict(X_test)
    print('\n',name, 'Mean Squared Error:', round(mean_squared_error(y_test, predictions),4))
    
    return regr

def trainModelKfold(name, X, y, seed):
    
    print(name)
    kf = KFold(n_splits=10, shuffle = True, random_state = seed)
    best = None
    bestModel = None
    X_test_best = None
    y_test_best = None
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        regr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        regr.fit(X_train, y_train)
        
        predictions = regr.predict(X_test)
        mse = round(mean_squared_error(y_test, predictions),4)
        
        if(best == None or best > mse):
            best = mse
            bestModel = regr
            X_test_best = X_test
            y_test_best = y_test
        print(i, 'Mean Squared Error:', mse)
        i+=1
    
    print('MSE:',round(mean_squared_error(y_test_best, bestModel.predict(X_test_best)),4))
    print('MAE:',round(mean_absolute_error(y_test_best, bestModel.predict(X_test_best)),4))
    print('R2 :',round(r2_score(y_test_best, bestModel.predict(X_test_best)),4))
    print('EVS:',round(explained_variance_score(y_test_best, bestModel.predict(X_test_best)),4))
    print('MXE:',round(max_error(y_test_best, bestModel.predict(X_test_best)),4))
    
    return bestModel

subfolder = 'Dataset/'
featuresdf = pd.read_pickle(subfolder+'pickle/more_exported_features_valence_arousal.pkl')
print('Vhodni podatki:')
print(featuresdf)

# genereSet = set(featuresdf['genere'].tolist())
# print(genereSet)

X_train = np.array(featuresdf['features'].tolist())

Y_valence = np.array(featuresdf['valence'].tolist())
Y_arousal = np.array(featuresdf['arousal'].tolist())

#displayVAgraph(Y_valence, Y_arousal, list(featuresdf.index) , 1, 9)

Y_valence_norm = normalizacija(Y_valence, 1, 1, 9)
Y_arousal_norm = normalizacija(Y_arousal, 1, 1, 9)

# displayVAgraph(Y_valence_norm, Y_arousal_norm, False, -1, 1)

print('\n')

model_valence = trainModelKfold('valence', X_train, Y_valence_norm, 0)
model_arousal = trainModelKfold('arousal', X_train, Y_arousal_norm, 0)

# save models
joblib.dump(model_valence, "./RFmodel/modelSVMValence.joblib")
joblib.dump(model_arousal, "./RFmodel/modelSVMArousal.joblib")