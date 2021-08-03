import sys
import os
import numpy as np
import pandas as pd

import pickle

# libraries for evaluation metrics:
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# model selection libraries:
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

# libraries for classifiers:
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# UDACITY libraries:
sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit

class evaluator:
    evas = []
    data_dict = {}
    feature_list = []

    def __init__(self, dict, selected_features, name) -> None:

        nb_analysis = {}
        dt_analysis = {}
        kn_analysis = {}

        self.data_dict = dict
        self.feature_list = selected_features

        nb_analysis = self.evaluate(GaussianNB())
        dt_analysis = self.evaluate(DecisionTreeClassifier())
        kn_analysis = self.evaluate(KNeighborsClassifier())

        self.evas.append(
            {
            name:{
                'GaussianNB': nb_analysis,
                'DecisionTreeClassifier': dt_analysis,
                'KNeighborsClassifier': kn_analysis
                }
            })

    def evaluate(self, classifier):

        eva_results = {}

        data = featureFormat(self.data_dict, self.feature_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        
        X = np.array(features)
        y = np.array(labels)

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        
        features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = classifier
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        
        if classifier == DecisionTreeClassifier():
            eva_results = {
                'Accuracy': accuracy_score(labels_test,pred),
                'Precision': precision_score(labels_test, pred, average='micro'),
                'Recall': recall_score(labels_test,pred),
                'Feature Importance': clf.feature_importances_
                }
        else:
            eva_results = {
                'Accuracy': accuracy_score(labels_test,pred),
                'Precision': precision_score(labels_test, pred, average='micro'),
                'Recall': recall_score(labels_test, pred, average='micro')
                }

        return eva_results
    
    def get_evas(self):
        return self.evas    

if __name__ == '__main__':
    data_dict = {}
    feature_list = []

    dataset_file = "./dataset/final_project_dataset.pkl"

    with open(dataset_file, "rb") as data_file:
        data_dict = pickle.load(data_file)

    feature_list = ['poi', 'salary', 'fraction_to_poi', 'from_messages']
    eva_obj = evaluator(data_dict, feature_list)
    
    gnb_dict = eva_obj.get_nb_eva
    print(gnb_dict)