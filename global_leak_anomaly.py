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
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import sklearn.metrics as metrics
import dataframe_image as dfi

def data_preprocessing(pressures, labels, test_size=0.25, scaling= True):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(pressures,labels, test_size=test_size, random_state=42)
    if scaling:    
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
    return x_train,x_test,y_train, y_test

def Isolation_forests_train_models(n_estimators_list, x_train):
    models = []
    for el in n_estimators_list:
        model = IsolationForest(n_estimators= el, contamination= 0.05, max_samples='auto', max_features=1.0)
        model.fit(x_train)
        model._name = "Iso_forest_" + str(el)
        models.append(model)
        print("train\t" + model._name + "\t finished")
    return models    

def Oneclass_SVM_train_models(x, y,parameters, verbose=True):
    models = []
    scores = []
    plt.figure(0).clf()
    for el in parameters:    
        nu = float(el.split("/")[0])
        gamma = el.split("/")[1]
        model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        model.fit(x)
        model._name = "OneclassSVM_g:"+ gamma + "_nu:" + str(nu)
        models.append(model)
        print("train\t" + model._name + "\t finished")
    return models


def local_outlier_factor_train_models(neighbors_number, x, y, verbose= True):
    models = []
    scores = []
    for el in neighbors_number :
        model = LocalOutlierFactor(n_neighbors = el, contamination=0.05, novelty=True)
        model.fit(x)
        model._name = "Local_outlier_" + str(el)
        models.append(model)
        print("train\t" + model._name + "\t finished")
    return models    

     
def evalutate_models(models, x, y, xs, ys, verbose= True ):
    scores = pd.DataFrame(columns=["models_name", "TN" ,"FP", "FN" ,"TP", "accuracy_score",
                                   "precision", "recall",
                                        "f1_score"]) 
    for i,model in enumerate(models):
        pred = []
        if "Iso_forest_" in model._name:
            pred =model.predict(xs)
        else:    
            pred =model.predict(x)
        pred = list(np.where(pred==-1, 0 , 1))
        accuracy_score = metrics.accuracy_score(y, pred)
        precision,recall,fscore, _ = metrics.precision_recall_fscore_support(y, pred)
        cm = metrics.confusion_matrix(y, pred)
        if verbose:
            print(model._name)
            print("Acc = " + str(accuracy_score) + "\t precision= " + str(precision[1])
                  + "\t recall=" + str(recall[1]) + "\tf1score= "+ str(fscore[1]))
            print(cm)
        tn= cm[0,0] 
        fp= cm[0,1] 
        fn= cm[1,0] 
        tp= cm[1,1] 
        scores.loc[i]= [str(model._name)] + [tn,fp,fn,tp] +[accuracy_score,precision[1],recall[1],fscore[1]]
    return scores
        
def export_results(scores, filename):
    pd.set_option("display.max_column", None)
    pd.set_option("display.max_colwidth", None)  
    pd.set_option('display.width', -1)
    pd.set_option('display.max_rows', None)
    styled_scores = scores.sort_values("accuracy_score",ascending=False).style.highlight_max(
        subset=["accuracy_score", "f1_score"])
    dfi.export(styled_scores,filename)      


#### load the Data

### Read all files in the Database
pressures, demands, flows, levels, leakages = read_file("Database/2018_SCADA")
junctions, pipes, cordinates, patterns = read_inp_file("Database/L-TOWN.inp")
data = water_network(pressures, demands, flows, levels,junctions, pipes,leakages,cordinates)

pressures = data.pressures.iloc[:,1:].to_numpy()
labels = np.sum(leakages.iloc[:, 1:].to_numpy(), axis=1)
#labels = [0 if i<=6.8 else 1 for i in labels]
labels = [0 if i==0 else 1 for i in labels]



""" Training isolation Forests """
x_train_s,x_test_s, y_train_s, y_test_s = data_preprocessing(pressures, labels, scaling = False)
#### First of all now we are going to use Isolation forests in order to find the best parameters in our network
#### Isolation forests and we specify the number of the trees
tree_numbers = [10,25,50,75,100,200,500,1000,10000]

iso= Isolation_forests_train_models(tree_numbers, x_train_s)




""" Training One class SVM """
x_train,x_test, y_train, y_test = data_preprocessing(pressures, labels)
svm_params = ["0.01/auto", "0.005/auto", "0.003/auto", "0.01/scale","0.005/scale","0.003/scale"]
svms = Oneclass_SVM_train_models(x_train, y_train, svm_params)



""" Training local outlier factor """
neighbors_number = [1,2,3,4,5,7,10,20,30]
lof = local_outlier_factor_train_models(neighbors_number, x_train, y_train, verbose=True)


models = iso + svms + lof

## evaluate train Data


print("Evaluating Train Data: \n\n")
results_train = evalutate_models(models,x_train, y_train, x_train_s, y_train_s)
print("======================================\n\n")    
export_results(results_train, "global_leak_anomaly_train.png")

## evaluate test Data
print("Evaluating Test Data: \n\n")
results_test = evalutate_models(models,x_test, y_test, x_test_s, y_test_s)
print("\n\n")
export_results(results_test, "global_leak_anomaly_test.png")
    
    
