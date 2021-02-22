import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt

#from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV #, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.losses import MSE, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, kl_divergence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

#Datensatz herunterladen
#Iris Datensatz: data_id=61
X,y = fetch_openml(data_id=61,return_X_y=True)

#Datensatz in Trainingssatz und Testsatz unterteilen
X_train, X_test, y_train, y_test = train_test_split(P,y, train_size=.8)


#Die Standardarchitektur des Modells festlegen
def build_nn(activation = 'relu', learning_rate = 3e-2, optimizer = SGD):
    model = Sequential()
    model.add(Dense(300,activation=activation))
    model.add(Dense(100,activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=categorical_crossentropy, 
                  optimizer= optimizer(learning_rate = learning_rate), 
                  metrics=["accuracy"])
    return model


#Klassifikator Pipeline
steps = [
            ('cleaning', SimpleImputer()),
            ('preprocessing', StandardScaler()),
            ('clf',  KerasClassifier(build_nn) )
        ]
pipe = Pipeline(steps=steps)

#Klassifikator trainieren
pipe.fit(X_train, y_train)

#Klassifikator testen
y_pred = pipe.predict(X_test)
accuracy_score(y_true=y_test,y_pred=y_pred)