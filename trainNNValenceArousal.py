import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
clear = lambda: os.system('cls')
clear()
print("\n"*5)

import numpy as np
import pandas as pd
import sklearn.preprocessing as skPre
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from pickle import dump

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.regularizers import l1
from keras.regularizers import l1_l2
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

from commonFunctions import normalizacija

np.set_printoptions(suppress=True)

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

def trainModel(X_train, X_test, y_train, y_test, seed, earlyStop=False, ):

    model = Sequential()  # , input_dim=193
    model.add(BatchNormalization())

    model.add(Dense( 500, activation='relu' ))
    #model.add(Dropout(rate = .2))
    model.add(Dense( 500, activation='relu' ))
    model.add(Dropout(rate = .2))
    model.add(Dense( 500, activation='relu' ))
    model.add(Dropout(rate = .2))
    model.add(Dense( 500, activation='relu' ))
    model.add(Dropout(rate = .2))
    model.add(Dense( 500, activation='relu' ))
    model.add(Dropout(rate = .2))
    model.add(Dense( 200, activation='relu' ))
    model.add(Dropout(rate = .2))
    model.add(Dense(   1, activation='linear' ))
    
    model.compile( loss = "mean_squared_error", optimizer = 'adam')
    
    if(earlyStop):
        early_callback = EarlyStopping(monitor='val_loss', patience=10 )
        history = model.fit( x = X_train, y = y_train, epochs=50, validation_data = (X_test, y_test) , verbose=0, callbacks=[early_callback] )
    else:
        history = model.fit( x = X_train, y = y_train, epochs=50, validation_data = (X_test, y_test) , verbose=0 )
    
    # evaluate the keras model
    loss = model.evaluate(X_test, y_test)
    #print('Validation loss: %.2f' % (loss))
    #showLoss(history)
    
    return model, history, loss
 
def normalizacija2(array, faktor):
    return 2.*(array - np.min(array))/np.ptp(array)-faktor


def trainModelKfold(name, X, y, seed):
    
    print(name)
    kf = KFold(n_splits=10, shuffle = True, random_state = seed)
    best = None
    bestModel = None
    bestHistory = None
    X_test_best = None
    y_test_best = None
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model, history, error = trainModel(X_train, X_test, y_train, y_test, seed, False)
        
        if(best == None or best > error):
            best = error
            bestModel = model
            bestHistory = history
            X_test_best = X_test
            y_test_best = y_test
        print(i, 'Mean Squared Error:', error)
        i+=1
    
    print('Best MSE:', best)
    
    print('MSE:',round(mean_squared_error(y_test_best, bestModel.predict(X_test_best)),4))
    print('MAE:',round(mean_absolute_error(y_test_best, bestModel.predict(X_test_best)),4))
    print('R2 :',round(r2_score(y_test_best, bestModel.predict(X_test_best)),4))
    print('EVS:',round(explained_variance_score(y_test_best, bestModel.predict(X_test_best)),4))
    print('MXE:',round(max_error(y_test_best, bestModel.predict(X_test_best)),4))
    
    return bestModel, bestHistory

subfolder = 'Dataset/'

featuresdf = pd.read_pickle(subfolder+'pickle/more_exported_features_valence_arousal.pkl')
#print(featuresdf)

#genereSet = set(featuresdf['genere'].tolist())
#print(genereSet)

X_train = np.array(featuresdf['features'].tolist())
Y_valence = np.array(featuresdf['valence'].tolist())
Y_arousal = np.array(featuresdf['arousal'].tolist())

Y_valence = normalizacija(Y_valence, 1, 1, 9)
Y_arousal = normalizacija(Y_arousal, 1, 1, 9)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
#model_valence, history_valence = trainModel(X_train, Y_valence_norm, 7, False)
#model_arousal, history_arousal = trainModel(X_train, Y_arousal_norm, 7, False)

model_valence, history_valence = trainModelKfold('valence', X_train, Y_valence, 0)
showLoss(history_valence)
model_valence.save('model_NN_10fold_valence')

model_arousal, history_arousal = trainModelKfold('arousal', X_train, Y_arousal, 0)
showLoss(history_arousal)
model_arousal.save('model_NN_10fold_arousal')