# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:00:55 2022

@author: Mohamed Amine Hammami
"""



from  read_data import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn import preprocessing
from tensorflow import keras
import tensorflow as tf
from sklearn.decomposition import PCA
from itertools import product
import dataframe_image as dfi
import sklearn.metrics as metrics


def data_preprocessing(pressures, labels, test_size=0.25):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(pressures,labels, test_size=test_size, random_state=42)
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
        
    return x_train,x_test,y_train, y_test


def models_init (
        layers_num,
        layer_sizes,
        input_shape,
        activation_hid_layer,
        activation_out_layer,
        ):
    layer_sizes = [layer_sizes] * layers_num
    layers_combinations = product(*layer_sizes)
    models = []
    #print(list(layers_combinations))
    for element in list(layers_combinations):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape = input_shape))

        model._name = element
        #print(element)
        for neurons_in_layer in element:
            model.add(tf.keras.layers.Dense(neurons_in_layer, activation = activation_hid_layer))
        model.add(tf.keras.layers.Dense(2, activation= activation_out_layer))
        models.append(model)
        #print(model.summary())
    return models    

    
    

def models_fit(
         x_train,
         y_traim,
         models,
         epochs=50,
         ):
     histories = []
     models_r = []
     print("training started")
     for model in models:
         model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'mse'])
    
         history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
         print("fit: \t" + str(model._name) + " \t finished")
         histories.append(history)
         models_r.append(model)
     return models_r, histories   


def evaluate_data(models, x, y, verbose= True ):
    scores = pd.DataFrame(columns=["models_name", "TN" ,"FP", "FN" ,"TP", "accuracy_score", "loss",
                                   "precision", "recall",
                                        "f1_score" ]) 
    for i,model in enumerate(models):
        score, acc, _ = model.evaluate(x, y, verbose=0)
        print("test " + str(model._name) + " finished with results loss_score: %f and acc %f" % (score,acc))
        pred = model.predict(x)
        pred_labels = [np.argmax(i) for i in pred]
        #pred = list(np.where(pred==-1, 0 , 1))
        accuracy_score = metrics.accuracy_score(y, pred_labels)
        precision,recall,fscore, _ = metrics.precision_recall_fscore_support(y, pred_labels)
        cm = metrics.confusion_matrix(y, pred_labels)
        if verbose:
            print(model._name)
            print("Acc = " + str(accuracy_score) + "\t precision= " + str(precision[1])
                  + "\t recall=" + str(recall[1]) + "\tf1score= "+ str(fscore[1]))
            print(cm)
        tn= cm[0,0] 
        fp= cm[0,1] 
        fn= cm[1,0] 
        tp= cm[1,1] 
        scores.loc[i]= [str(model._name)] + [tn,fp,fn,tp] +[accuracy_score,score,precision[1],recall[1],fscore[1]]
    return scores    
 
    
def export_results(scores, filename):
     pd.set_option("display.max_column", None)
     pd.set_option("display.max_colwidth", None)  
     pd.set_option('display.width', -1)
     pd.set_option('display.max_rows', None)
     styled_scores = scores.sort_values("accuracy_score",ascending=False).style.highlight_max(
         subset=["accuracy_score", "f1_score"])
     dfi.export(styled_scores,filename)
     
     
### Read all files in the Database
pressures, demands, flows, levels, leakages = read_file("Database/2018_SCADA")
junctions, pipes, cordinates, patterns = read_inp_file("Database/L-TOWN.inp")
data = water_network(pressures, demands, flows, levels,junctions, pipes,leakages,cordinates)

demands_cols = np.array(data.demands.columns)
pressures_cols = np.array(data.pressures.columns)


relevant_pipes= []
for index,pipe in data.pipes.iterrows():
    if pipe['Node1'] in demands_cols or pipe['Node2'] in demands_cols:
        #print(pipe)
        relevant_pipes.append([index,pipe['Node1'],pipe['Node2']])
print(np.shape(relevant_pipes)) 
relevant_pipes_name = np.array(relevant_pipes)[:,0]

leakages = data.leakages.iloc[:, 1:6:4].to_numpy()
labels = np.sum(leakages, axis=1)
labels = [0 if i==0 else 1 for i in labels]

pressures_c = []
for idx,el in enumerate(pressures_cols):
    if el in demands_cols and "Timestamp" not in el:
        pressures_c.append(data.pressures.iloc[:,idx].to_numpy())
pressures_c = np.array(pressures_c).T



x_train, x_test, y_train,y_test = data_preprocessing(pressures_c, labels)



#### now initialise the models which we want to test 
layer_sizes = [30,50,70,100,300]
input_shape = (np.shape(x_train)[1],)
models= models_init(2,layer_sizes,input_shape,'relu','sigmoid')


#### fit the data
models_f, histories = models_fit(x_train, y_train, models)
scores = evaluate_data(models_f, x_test,y_test)


export_results(scores, "local_c_leak_NN.png")