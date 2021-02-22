# Hyperparameter Optimization

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
X,y = fetch_openml(data_id=554,return_X_y=True)

#Datensatz in Trainingssatz und Testsatz unterteilen
X_train, X_test, y_train, y_test = train_test_split(P,y, train_size=.8)


#Die Standardarchitektur des Modells festlegen
def build_nn(activation = 'relu', 
             learning_rate = 3e-2, 
             optimizer = SGD):
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
            ('preprocessing', StandardScaler()),
            ('clf',  KerasClassifier(build_nn) )
        ]
pipe = Pipeline(steps=steps)

#Klassifikator trainieren
pipe.fit(X_train, y_train)

#Klassifikator testen
y_pred = pipe.predict(X_test)
accuracy_score(y_true=y_test,y_pred=y_pred)

#Suchraum definieren
params = {
          'clf__activation': ['linear', 'sigmoid', 'tanh', 'relu'],
          'clf__learning_rate': [1e-4, 1e-3, 1e-2, 0.1],
          'clf__optimizer': [Adam, SGD] 
         }

#Grid Search
clf1 = GridSearchCV(estimator=pipe, 
                    param_grid=params, 
                    cv=4, 
                    error_score=.0, 
                    verbose=2)

clf1.fit(X=X_train,y=y_train)

#Predict:
y_pred = clf1.predict(X_test)
accuracy_score(y_true=y_test,y_pred=y_pred)

#Best Scores + Params
clf1.best_score_, clf1.best_params_



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

# Evolutionary Search
clf3 = EvolutionaryAlgorithmSearchCV(estimator=pipe, params=params,
                                     scoring='accuracy', cv=4, verbose=2,
                                     population_size=50, gene_mutation_prob=0.10,
                                     gene_crossover_prob=.5, tournament_size=3,
                                     generations_number=5, n_jobs=1)

clf3.fit(X=X_train,y=y_train)

#Predict:
y_pred = clf3.predict(X_test)
accuracy_score(y_true=y_test,y_pred=y_pred)

#Best Scores + Params
clf3.best_score_, clf3.best_params_


#Ergebnisse darstellen
def plot_results(index='dar__ordar', columns='dar__ordriv'):
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
    plt.savefig('Randomized_Search.png')
    plt.show()
    


#plot_results(index='clf__activation', columns='clf__learning_rate')
plot_results(index='clf__learning_rate', columns='clf__activation')












