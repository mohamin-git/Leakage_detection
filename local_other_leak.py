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
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import dataframe_image as dfi
from sklearn.utils import resample


    
def data_preprocessing(p_nodes, l_pipes, test_size=0.25, scaling = False):
    labels = []
    for leakage in l_pipes:
        labels.append(data.leakages.loc[:,leakage].to_numpy())
    labels = np.sum(np.array(labels), axis=0)
    labels = np.where(labels>0, 1, 0)
    pressures = []
    for node in p_nodes:
        pressures.append(data.pressures.loc[:,node].to_numpy())
    pressures = np.array(pressures).T
    #print(pressures)
    print(pressures.shape)
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(pressures,labels, test_size=test_size, random_state=42)
    if scaling == True:
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    return x_train,x_test,y_train, y_test

 
def random_Forest_train_models(n_estimators_list, x_train, y_train):
    models = []
    for el in n_estimators_list:
        model = RandomForestClassifier(n_estimators= el)
        #sample_weight = [35 if y==1 else 1  for y in y_train]
        model.fit(x_train, y_train)
        model._name = "Rand_forest_" + str(el)
        models.append(model)
        print("train\t" + model._name + "\t finished")
    return models  

def evaluate_data(models, x, y, verbose= True, nn =False ):
    scores = pd.DataFrame(columns=["models_name", "TN" ,"FP", "FN" ,"TP", "accuracy_score",
                                   "precision", "recall",
                                        "f1_score", "AUC" ]) 
    plt.figure(0).clf()
    for i,model in enumerate(models):
        pred = model.predict(x)
        #print(len(pred))
        #pred = list(np.where(pred==-1, 0 , 1))
        if nn:
            pred = [np.argmax(i) for i in pred]
        accuracy_score = metrics.accuracy_score(y, pred)
        precision,recall,fscore, _ = metrics.precision_recall_fscore_support(y, pred)
        cm = metrics.confusion_matrix(y, pred)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr,tpr,label= str(model._name)+ " AUC=" +str(auc))
        if verbose:
            print(model._name)
            print("Acc = " + str(accuracy_score) + "\t precision= " + str(precision[1])
                  + "\t recall=" + str(recall[1]) + "\tf1score= "+ str(fscore[1]))
            print(cm)
        tn= cm[0,0] 
        fp= cm[0,1] 
        fn= cm[1,0] 
        tp= cm[1,1] 
        scores.loc[i]= [str(model._name)] + [tn,fp,fn,tp] +[accuracy_score,precision[1],recall[1],fscore[1], auc]
    plt.legend()
    plt.savefig("local_leak_randomF_AUC.png")
    return scores  
 
def nn_models_init (
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

    
    

def nn_models_fit(
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

    
def export_results(scores, filename):
     pd.set_option("display.max_column", None)
     pd.set_option("display.max_colwidth", None)  
     pd.set_option('display.width', -1)
     pd.set_option('display.max_rows', None)
     styled_scores = scores.sort_values("accuracy_score",ascending=False).style.highlight_max(
         subset=["accuracy_score", "f1_score"])
     dfi.export(styled_scores,filename)
     
     
def data_region(pipe, distance):
    #search pipe_cordinates 
    n1,n2 = data.pipes.loc[pipe]['Node1'],data.pipes.loc[pipe]['Node2']
    #n1,n2 = int(n1.replace("n","")),int(n2.replace("n",""))
    x1,y1 = float(data.cordinates.loc[n1]['X-Coord']), float(data.cordinates.loc[n1]['Y-Coord'])
    x2,y2 = float(data.cordinates.loc[n2]['X-Coord']), float(data.cordinates.loc[n2]['Y-Coord'])
    xp,yp = (x1+x2)/2 , (y1+y2)/2
    nodes = []
    pipes = []
    for id, node in data.cordinates.iterrows():
        if np.linalg.norm(np.array([xp, yp]) - np.array([float(node['X-Coord']), float(node['Y-Coord'])])) <= distance:
            nodes.append(id)
    for id, pipe in data.pipes.iterrows():
        if pipe["Node1"] in nodes and pipe["Node2"] in nodes:
            pipes.append(id)
    l_pipes= np.intersect1d(pipes, data.leakages.columns)
    p_nodes= np.intersect1d(nodes, data.pressures.columns)
    #print(xp,yp)
    assert len(l_pipes)>0, "there is no data, the chosen region is too small"
    assert len(p_nodes)>0, "there is no data, the chosen region is too small"
    return p_nodes, l_pipes

     
     
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


"""" Get all pressures and leaks information from pipe p1 in radius of  300m"""
a,b = data_region('p1',300)
print(a,b)


"""  Train a Random Forest classifier on collected Data and export results"""
x_train, x_test, y_train,y_test = data_preprocessing(a, b)
tree_numbers = [10,25,50,75,100]
#### fit the model to the data
models_f= random_Forest_train_models(tree_numbers, x_train, y_train)
scores = evaluate_data(models_f, x_test,y_test)
export_results(scores, "local_leak_randomF.png")


""" Train a Random Forest classifier on collected Data and export results"""
x_train, x_test, y_train,y_test = data_preprocessing(a, b, scaling =True)
layer_sizes = [30,50,100]
input_shape = (np.shape(x_train)[1],)
models= nn_models_init(2,layer_sizes,input_shape,'relu','sigmoid')


#### fit the data
models_f, histories = nn_models_fit(x_train, y_train, models)
scores = evaluate_data(models_f, x_test,y_test, nn=True)
export_results(scores, "local_leak_NN.png")






