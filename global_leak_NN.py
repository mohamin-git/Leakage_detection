# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:09:38 2022

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


# scale the data before training
def data_preprocessing(pressures, labels, test_size=0.25):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(pressures,labels, test_size=test_size, random_state=42)
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
        
    return x_train,x_test,y_train, y_test


# initialize all models
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
 
    
def models_test(
         models,
         x_test,
         y_test,
         save_img = True,
         files_name = []):
    scores = pd.DataFrame(columns=["models_name", "TN", "FP", "FN", "TP", "accuracy_score",
                                    "loss_score", "precision", "recall",
                                    "f1_score" ]) 
    for i,model in enumerate(models):
        #print(model.metrics_names)
        score, acc, _ = model.evaluate(x_test, y_test, verbose=0)
        print("test " + str(model._name) + " finished with results loss_score: %f and acc %f" % (score,acc))
        y_predicted = model.predict(x_test)
        y_predicted_labels = [np.argmax(i) for i in y_predicted]
        sess = tf.Session()
        confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels, num_classes=2)
        #print(confusion_matrix)
        with sess:
            cm = confusion_matrix.eval()
            #print("confusion Matrix: \n", cm)
        recall = cm [1,1] / sum(cm[1])
        #print("Recall(TPR): \t", recall)
        precision = cm [0,0] / sum(cm[0])
        #print("Precison(TNR): \t", selectivity)
        f1 = 2 * (precision * recall) / (precision + recall)
        tn= cm[0,0] 
        fp= cm[0,1] 
        fn= cm[1,0] 
        tp= cm[1,1] 
        scores.loc[i]= [str(model._name)] + [tn,fp,fn,tp] +[acc,score,precision,recall,f1]
    pd.set_option("display.max_column", None)
    pd.set_option("display.max_colwidth", None)  
    pd.set_option('display.width', -1)
    pd.set_option('display.max_rows', None)
    styled_scores = scores.style.highlight_max(
        subset=["accuracy_score", "f1_score"]).highlight_min(
            subset=["loss_score"])
    if save_img:
        dfi.export(styled_scores,files_name[0])
        styled_scores = scores.sort_values("accuracy_score",ascending=False).style.highlight_max(
            subset=["accuracy_score", "f1_score"]).highlight_min(
                subset=["loss_score"])
        dfi.export(styled_scores,files_name[1])
        styled_scores = scores.sort_values("f1_score",ascending=False).style.highlight_max(
            subset=["accuracy_score", "f1_score"]).highlight_min(
                subset=["loss_score"])
        dfi.export(styled_scores,files_name[2])
        
        
    return models,scores  


### Read all files in the Database
pressures, demands, flows, levels, leakages = read_file("Database/2018_SCADA")
junctions, pipes, cordinates, patterns  = read_inp_file("Database/L-TOWN.inp")
data = water_network(pressures, demands, flows, levels,junctions, pipes,leakages,cordinates)




"""
    In this section, we will try to train the full Dataset without 
    Downsampling 

"""
#### prepare the data and preprocessing
pressures = data.pressures.iloc[:,1:].to_numpy()
labels = np.sum(leakages.iloc[:, 1:].to_numpy(), axis=1)
labels = [0 if i==0 else 1 for i in labels] 
x_train, x_test, y_train,y_test = data_preprocessing(pressures, labels)



#### now initialise the models which we want to test 
layer_sizes = [30,50,70,100,300]
input_shape = (np.shape(x_train)[1],)
models= models_init(2,layer_sizes,input_shape,'relu','sigmoid')


#### fit the data
models_f, histories = models_fit(x_train, y_train, models)



#### now we test the data
files_name = ["global_leak_NN.png","global_leak_NN_acc.png",
               "global_leak_NN_f1.png"]
# The results are going to be saved  in png file
models_f, scores = models_test(models_f,x_test,y_test, files_name = files_name)



"""
    In this section, we will try to downsample the Data, that means we will average 
    the sensor Data to every hour 
"""

### now we average the pressures for every hour and we create the new Label
pressures_h = []
arr = np.arange(0,len(labels), 12)
for idx,el in enumerate(arr[:-1]):
    pressures_h.append(np.mean(data.pressures.iloc[arr[idx]:arr[idx+1], 1:].to_numpy(), axis=0))
labels_h = []
for idx,el in enumerate(arr[:-1]):
    labels_h.append(1 if np.mean(labels[arr[idx]:arr[idx+1]]) > 0.5 else 0)
    

#### prepare the data and preprocessing
x_train, x_test, y_train,y_test = data_preprocessing(pressures_h, labels_h) 


#### now initialise the models which we want to test 
layer_sizes = [30,50,70,100,300]
input_shape = (np.shape(x_train)[1],)
models_h= models_init(2,layer_sizes,input_shape,'relu','sigmoid')


#### fit the data
models_hf, histories = models_fit(x_train, y_train, models_h)

#### now we test the data
files_name = ["global_leak_NN_h.png","global_leak_NN_h_acc.png",
               "global_leak_NN_h_f1.png"]
models_hf, scores_h = models_test(models_hf,x_test,y_test, files_name = files_name)   




"""
test the Network on 2019 dataset 


pressures, demands, flows, levels, leakages = read_file("Database/2019_SCADA")
junctions, pipes, cordinates = read_inp_file("Database/L-TOWN.inp")
data_19 = water_network(pressures, demands, flows, levels,junctions, pipes,leakages,cordinates)

#### prepare the data and preprocessing
pressures_19 = data_19.pressures.iloc[:,1:].to_numpy()
labels_19 = np.sum(data_19.leakages.iloc[:, 1:].to_numpy(), axis=1)
labels_19 = [0 if i==0 else 1 for i in labels_19] 

"""