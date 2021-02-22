# Neural Architecture Search CNN
## Pipelines

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
#MNIST Datensatz: data_id=554
#Fashion-MNIST Datensatz: data_id=40996
#CIFAR-10 Datensatz: data_id=40927
X,y = fetch_openml(data_id=554,return_X_y=True)

#Wenn es sich um ein Bild mit drei Kanäle (Farbbild) handelt, dann Farbe=True, sonst Farbe=False
farbe = False

#Preprocessing
if farbe:
    P=X
else:
    scaler = StandardScaler()
    P = scaler.fit_transform(X)
    
size = list(X.shape)
def Farbe(XX, farbe, Bild_size):
    if farbe:
        X = XX.reshape(len(y), 3, np.sqrt(Bild_size/3).astype(int), np.sqrt(Bild_size/3).astype(int)).transpose(0,2,3,1).astype('int32')
    else:
        X = XX.reshape((len(y), 1, np.sqrt(Bild_size).astype(int), np.sqrt(Bild_size).astype(int))).transpose(0,2,3,1).astype('int32')
    return X


P = Farbe(XX = P, farbe = farbe, Bild_size = size[1])
y = y.astype(int)

#Datensatz in Trainingssatz und Testsatz unterteilen
X_train, X_test, y_train, y_test = train_test_split(P,y, train_size=.8)

#Vorschau von dem Bild
for i in range(10, 15):
    some_digit_train = X_train[i]
    plt.imshow(some_digit_train, cmap="binary")
    plt.title(y_train[i])
    plt.show()

#Input_Shape von dem Bild
if farbe:
    input_shape = list(P.shape)
    input_shape = (input_shape[1],input_shape[2],input_shape[3])
else:
    input_shape = list(P.shape)
    input_shape = (input_shape[1],input_shape[2],1)

#Die Standardarchitektur des Modells festlegen
def build_nn(activation = 'relu', 
             activation_end='softmax', 
             n_neurons_1=200, 
             n_neurons_2=100,
             n_conv=1,
             kernel_size=3,
             input_shape = input_shape,
             n_layers=1):

        model = Sequential()
        model.add(Conv2D(filters = 32, kernel_size=(kernel_size, kernel_size), padding="same", activation=activation, input_shape=input_shape, name="conv_1"))
        model.add(MaxPooling2D(pool_size=(2, 2), name="Max_Pooling_1"))
        for j in range(n_conv):
            model.add(Conv2D(filters = 32*2**(j), kernel_size=(kernel_size, kernel_size), padding="same", activation=activation))        
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        for i in range(n_layers):
            model.add(Dense(n_neurons_1,activation=activation , name="Schicht_1"))
        model.add(Dense(n_neurons_2,activation=activation, name="Schicht_2"))
        model.add(Dense(10,activation=activation_end, name="Output"))

        lr_schedule = ExponentialDecay(
        initial_learning_rate = 0.001,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

        model.compile(loss=sparse_categorical_crossentropy, 
                      optimizer= Adam(learning_rate = lr_schedule), 
                      metrics=["accuracy"])
        return model

#Ansicht des Modells
model = build_nn()
model.summary()

#Klassifikator Pipeline
steps = [
    ('clf',  KerasClassifier(build_nn) )
]
pipe = Pipeline(steps=steps)

#Klassifikator trainieren
pipe.fit(X_train, y_train)

#Klassifikator testen
y_pred = pipe.predict(X_test)
accuracy_score(y_true=y_test,y_pred=y_pred)

#Suchraum definieren
params = [
            {
            'clf__n_conv': [1, 2, 3, 4],
            'clf__activation': ['linear','tanh', 'relu', 'sigmoid'],
            'clf__n_layers': [0, 1, 2, 3],
            'clf__n_neurons_1': [400, 600, 800, 1000],
            'clf__n_neurons_2': [50, 100, 200, 300]
            }    
    ]

# Randomized Search
clf2 = RandomizedSearchCV(estimator=pipe, 
                          param_distributions=params, 
                          n_iter=300, 
                          cv=5, 
                          verbose=2)

clf2.fit(X=X_train,y=y_train)


#Predict:
y_pred = clf2.predict(X_test)
accuracy_score(y_true=y_test,y_pred=y_pred)

#Best Scores + Params
clf2.best_score_, clf2.best_params_

#Ergebnisse darstellen
def plot_results_1(index='dar__ordar', columns='dar__ordriv'):
    """Select two hyperparameters from which we plot the fluctuations"""
    index = 'param_' + index
    columns = 'param_' + columns

    # prepare the results into a pandas.DataFrame
    df = pd.DataFrame(clf2.cv_results_)

    # Remove the other by selecting their best values (from gscv.best_params_)
    other = [c for c in df.columns if c[:6] == 'param_']
    other.remove(index)
    other.remove(columns)
    for col in other:
        df = df[df[col] == clf2.best_params_[col[6:]]]

    # Create pivot tables for easy plotting
    table_mean = df.pivot_table(index=index, columns=columns,
                                values=['mean_test_score'])
    table_std = df.pivot_table(index=index, columns=columns,
                               values=['std_test_score'])

    # plot the pivot tables
    plt.figure()
    ax = plt.gca()
    for col_mean, col_std in zip(table_mean.columns, table_std.columns):
        table_mean[col_mean].plot(ax=ax, yerr=table_std[col_std], marker='o',
                                  label=col_mean)
    plt.title('Randomized-search results (higher is better)')
    plt.ylabel('Accuracy Score')
    plt.legend(title=table_mean.columns.names)
    plt.savefig('Randomized_Search_1.png')
    plt.show()

#plot_results(index='clf__activation', columns='clf__learning_rate')
plot_results_1(index='clf__n_conv', columns='clf__activation')

def plot_results_2(index='dar__ordar', columns='dar__ordriv'):
    """Select two hyperparameters from which we plot the fluctuations"""
    index = 'param_' + index
    columns = 'param_' + columns

    # prepare the results into a pandas.DataFrame
    df = pd.DataFrame(clf2.cv_results_)

    # Remove the other by selecting their best values (from gscv.best_params_)
    other = [c for c in df.columns if c[:6] == 'param_']
    other.remove(index)
    other.remove(columns)
    for col in other:
        df = df[df[col] == clf2.best_params_[col[6:]]]

    # Create pivot tables for easy plotting
    table_mean = df.pivot_table(index=index, columns=columns,
                                values=['mean_test_score'])
    table_std = df.pivot_table(index=index, columns=columns,
                               values=['std_test_score'])

    # plot the pivot tables
    plt.figure()
    ax = plt.gca()
    for col_mean, col_std in zip(table_mean.columns, table_std.columns):
        table_mean[col_mean].plot(ax=ax, yerr=table_std[col_std], marker='o',
                                  label=col_mean)
    plt.title('Randomized-search results (higher is better)')
    plt.ylabel('Accuracy Score')
    plt.legend(title=table_mean.columns.names)
    plt.savefig('Randomized_Search_2.png')
    plt.show()
    
plot_results_2(index='clf__n_neurons_1', columns='clf__activation')

def plot_results_3(index='dar__ordar', columns='dar__ordriv'):
    """Select two hyperparameters from which we plot the fluctuations"""
    index = 'param_' + index
    columns = 'param_' + columns

    # prepare the results into a pandas.DataFrame
    df = pd.DataFrame(clf2.cv_results_)

    # Remove the other by selecting their best values (from gscv.best_params_)
    other = [c for c in df.columns if c[:6] == 'param_']
    other.remove(index)
    other.remove(columns)
    for col in other:
        df = df[df[col] == clf2.best_params_[col[6:]]]

    # Create pivot tables for easy plotting
    table_mean = df.pivot_table(index=index, columns=columns,
                                values=['mean_test_score'])
    table_std = df.pivot_table(index=index, columns=columns,
                               values=['std_test_score'])

    # plot the pivot tables
    plt.figure()
    ax = plt.gca()
    for col_mean, col_std in zip(table_mean.columns, table_std.columns):
        table_mean[col_mean].plot(ax=ax, yerr=table_std[col_std], marker='o',
                                  label=col_mean)
    plt.title('Randomized-search results (higher is better)')
    plt.ylabel('Accuracy Score')
    plt.legend(title=table_mean.columns.names)
    plt.savefig('Randomized_Search_3.png')
    plt.show()
    
plot_results_3(index='clf__n_layers', columns='clf__activation')


def plot_results_4(index='dar__ordar', columns='dar__ordriv'):
    """Select two hyperparameters from which we plot the fluctuations"""
    index = 'param_' + index
    columns = 'param_' + columns

    # prepare the results into a pandas.DataFrame
    df = pd.DataFrame(clf2.cv_results_)

    # Remove the other by selecting their best values (from gscv.best_params_)
    other = [c for c in df.columns if c[:6] == 'param_']
    other.remove(index)
    other.remove(columns)
    for col in other:
        df = df[df[col] == clf2.best_params_[col[6:]]]

    # Create pivot tables for easy plotting
    table_mean = df.pivot_table(index=index, columns=columns,
                                values=['mean_test_score'])
    table_std = df.pivot_table(index=index, columns=columns,
                               values=['std_test_score'])

    # plot the pivot tables
    plt.figure()
    ax = plt.gca()
    for col_mean, col_std in zip(table_mean.columns, table_std.columns):
        table_mean[col_mean].plot(ax=ax, yerr=table_std[col_std], marker='o',
                                  label=col_mean)
    plt.title('Randomized-search results (higher is better)')
    plt.ylabel('Accuracy Score')
    plt.legend(title=table_mean.columns.names)
    plt.savefig('Randomized_Search_4.png')
    plt.show()
    
plot_results_4(index='clf__n_neurons_2', columns='clf__activation')












