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
    
    evas = {}
    data_dict = {}
    feature_list = []

    def __init__(self, dict, selected_features=None, name=None) -> None:

        self.data_dict = dict

        if selected_features!=None and name!=None:
            self.add_eva(selected_features, name)

    def add_eva(self, selected_features, name):

        nb_analysis = {}
        dt_analysis = {}
        kn_analysis = {}

        self.feature_list = selected_features

        nb_analysis = self.evaluate(GaussianNB())
        dt_analysis = self.evaluate(DecisionTreeClassifier())
        kn_analysis = self.evaluate(KNeighborsClassifier())

        self.evas[name] = {
                'GaussianNB': nb_analysis,
                'DecisionTreeClassifier': dt_analysis,
                'KNeighborsClassifier': kn_analysis
                }

    def evaluate(self, classifier):

        eva_results = {}

        data = featureFormat(self.data_dict, self.feature_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        
        X = np.array(features)
        y = np.array(labels)

        # sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=42)
        
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


    def to_dataframe(self):

        eva_nested_dict = self.evas
        dfs = []
        dfs.append(pd.DataFrame.from_dict({(i,j): eva_nested_dict[i][j] 
                        for i in eva_nested_dict.keys() 
                        for j in eva_nested_dict[i].keys()},
                    orient='index'))
        return dfs

if __name__ == '__main__':
    data_dict = {}
    dataset_file = "dataset/final_project_dataset.pkl"

    with open(dataset_file, "rb") as data_file:
        data_dict = pickle.load(data_file)

    eva_list_1 = ['poi','salary','bonus','from_messages']
    eva_list_2 = ['poi','bonus','exercised_stock_options','from_messages']
    
    eva_obj = evaluator(data_dict)

    eva_obj.add_eva(eva_list_1, 'eva_set_1')
    eva_obj.add_eva(eva_list_2, 'eva_set_2')

    evas = eva_obj.get_evas()
    dfs = eva_obj.to_dataframe()
    print(dfs)