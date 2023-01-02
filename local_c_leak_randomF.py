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



def data_undersampling(pressures, labels, pressures_cols, demands_cols):
    idxs = []
    for idx,el in enumerate(pressures_cols):
        if el in demands_cols and "Timestamp" not in el:
            idxs.append(idx)  
            
    pressures_c = pressures.iloc[:,idxs]
    pressures_c.reset_index(drop=True, inplace=True)
    pressures_c.loc[:, "labels"] = list(labels)
    pressures_c_0 = pressures_c[pressures_c["labels"] == 0]
    pressures_c_1 = pressures_c[pressures_c["labels"]  == 1]
    pressures_c_1_under = pressures_c_1.sample(len(pressures_c_0))
    pressures_c_under = pd.concat([pressures_c_0, pressures_c_1_under])
    labels_under = np.concatenate((np.zeros(len(pressures_c_0)), np.ones(len(pressures_c_0))))
    pressures_c_under = pressures_c_under.to_numpy()[:,:-1]
    return pressures_c_0, pressures_c_1_under, pressures_c_under, labels_under

    
    
def train_test_split(pressures, labels, test_size=0.25):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(pressures,labels, test_size=test_size, random_state=42)
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

def evaluate_data(models, x, y, verbose= True ):
    scores = pd.DataFrame(columns=["models_name", "TN" ,"FP", "FN" ,"TP", "accuracy_score",
                                   "precision", "recall",
                                        "f1_score", "AUC" ]) 
    plt.figure(0).clf()
    for i,model in enumerate(models):
        pred = model.predict(x)
        #print(len(pred))
        #pred = list(np.where(pred==-1, 0 , 1))
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
    plt.savefig("local_c_leak_randomF_AUC.png")
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


pressures_c_0, pressures_c_1_under, pressures_c_under, labels_under = data_undersampling(data.pressures, labels, pressures_cols, demands_cols)


x_train, x_test, y_train,y_test = train_test_split(pressures_c_under, labels_under)

tree_numbers = [10,25,50,75,100,200,500,1000,10000]
#### fit the model to the data
models_f= random_Forest_train_models(tree_numbers, x_train, y_train)
scores = evaluate_data(models_f, x_test,y_test)
export_results(scores, "local_c_leak_randomF.png")



## plot 3d scatter plot to visualize region c 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(len(labels_under)//2):
    ax.scatter(pressures_c_0.iloc[i,0], pressures_c_0.iloc[i,1], pressures_c_0.iloc[i,2], c="g")
    ax.scatter(pressures_c_1_under.iloc[i,0], pressures_c_1_under.iloc[i,1], pressures_c_1_under.iloc[i,2], c="r")
    
plt.savefig("region_c_scatter.png")
 