

class evaluator:
    
    import sys
    import os

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
    sys.path.append("../tools/")
    from feature_format import featureFormat, targetFeatureSplit

    data_dict = {}
    nb_analysis = {}
    dt_analysis = {}
    kn_analysis = {}

    def __init__(self, dict, selected_features) -> None:
        self.data_dict = dict

        self.nb_analysis = self.evaluator(GaussianNB(), selected_features, dict)
        self.dt_analysis = self.evaluator(DecisionTreeClassifier(), selected_features, dict)
        self.kn_analysis = self.evaluator(KNeighborsClassifier(), selected_features, dict)

    def evaluator(self, clasifier, features):
        data = featureFormat(my_dataset, feature_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)

        
        X = np.array(features)
        y = np.array(labels)

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        
        features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = classifier
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        
        if classifier == DecisionTreeClassifier():
            return {'Accuracy': accuracy_score(labels_test,pred),'Precision': precision_score(labels_test,pred, average='micro'),
                    'Recall': recall_score(labels_test,pred), 'Feature Importance': clf.feature_importances_}
        
        return {'Accuracy': accuracy_score(labels_test,pred),'Precision': precision_score(labels_test,pred, average='micro'),
                'Recall': recall_score(labels_test,pred, average='micro')}