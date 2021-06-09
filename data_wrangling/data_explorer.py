import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append("../data_wrangling/")

class explorer:
    """
        Class for exploring the Enron dataset (emails + financial data);
        loads up the dataset (pickled dict of dicts).

        Attributes
        ----------
        enron_data  :   dict
            a dictionary with all enron data with the form:
            enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }
            where {features_dict} is a dictionary of features associated with that person.
            example: enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
        enron_df    :   pandas dataframe
            a Pandas DataFrame with all enron data

        Methods
        -------
        get_number_of_entries()
            Returns number of entries in the dataset

        References:
        -------
    """
    
    pois = []
    nans = []
    clean_data = []
    enron_data = {}
    enron_df = None
    params = {}
    features = []
    feature_nan_count = {}

    def __init__(self, data_dict):        
        '''
        Parameters considered:
        ---------------------
            'Features'
            'Features with NaN'
            'Number of individuals'
            'Names'
            'Number of POIs'
            'POI list'
            'Number of non POIs'
            'Salary'
            'Folks with salary'
            'Number of Folks with NaN salary'
            'Percentage of Folks with NaN salary (%)'
            'Number of POIs with NaN salary'
            'Percentage of POIs with NaN salary (%)'
        '''
        
        self.list_of_entries_with_nan(data_dict)
        self.params['Features with NaN'] = self.nans
        self.enron_data = data_dict
        self.enron_df = pd.DataFrame.from_dict(self.enron_data).transpose()        
        self.enron_df = self.enron_df.replace('NaN', pd.np.nan)
        self.enron_df['Name'] = self.enron_df.index
        self.features = self.enron_df.columns.tolist()

        df = self.enron_df.copy()

        # print(self.enron_df)
        # How many people is in the database
        n_folks = self.df_size(self.enron_df)
        self.params['Number of individuals'] = n_folks
        self.params['Names'] = df['Name'].tolist()

        # How many attributes for each person:
        self.params['Number of features'] = len(df.columns)

        # How many are POIs?
        n_pois = self.df_size(df[df.poi.eq(True)])
        self.params['Number of POIs'] = n_pois
        df_poi = df[df.poi.eq(True)]
        # print(df_poi.index)
        self.pois = df_poi.index.tolist()
        self.params['POI list'] = self.pois
        # Number of non-POIs:
        self.params['Number of non POIs'] = n_folks - n_pois

        # How many folks in this dataset have a quantified salary?
        df_sal = df['salary'].dropna() #subset=['salary'])
        n_folks_with_salary = self.df_size(df_sal)
        self.params['Folks with salary'] = n_folks_with_salary

        # How many people in the E+F dataset (as it currently exists) have “NaN”
        # for their total payments? What percentage of people in the dataset as 
        # a whole is this?
        n_folks_with_nan_pay = self.enron_df['total_payments'].isnull().sum()
        percent_folks_with_nan_pay = 0
        if n_folks > 0:
            percent_folks_with_nan_pay = n_folks_with_nan_pay/n_folks*100
        
        self.params['Number of Folks with NaN salary'] = n_folks_with_nan_pay
        self.params['Percentage of Folks with NaN salary (%)'] = percent_folks_with_nan_pay

        # How many POIs in the E+F dataset have “NaN” for their total payments? 
        # What percentage of POI’s as a whole is this?
        df = self.enron_df
        df_poi = df[df.poi.eq(True)]

        n_poi = self.df_size(df_poi)
        n_poi_with_nan_pay = df_poi['total_payments'].isnull().sum()

        percent_pois_with_nan_pay = 0
        if n_poi > 0:
            percent_pois_with_nan_pay = n_poi_with_nan_pay/n_poi*100
         
        self.params['Number of POIs with NaN salary'] = n_poi_with_nan_pay
        self.params['Percentage of POIs with NaN salary (%)'] = percent_pois_with_nan_pay

        self.calc_missing_data()

    def get_dataset_parameters(self):
        '''Return parameters from dataset'''
        return self.params

    def df_size(self, df):
        '''Return number of rows in Pandas DataFrame'''
        return (df.shape)[0]

    def get_dataframe(self):
        return self.enron_df

    def list_of_entries_with_nan(self, data_dict):       

        nans_dict = {}
        #--- for each feature sum NaN values
        for employee_name, attributes_dict in data_dict.items():
            nan_count = 0
            nans_dict = attributes_dict.copy()
            nans_dict['employee_name'] = employee_name
            nans_dict['nan_count'] = nan_count

            for feature_name, feature_value in attributes_dict.items():
                if str(feature_value).lower() == 'nan':
                    nans_dict['nan_count'] += 1

            if nans_dict['nan_count'] > 0:
                self.nans.append(nans_dict)
                
    def get_poi_list(self):
        return self.pois

    def get_nan_list(self):
        return self.nans

    def number_of_pois(self):
        return len(self.pois)

    def get_feature_nan_counts(self):
        return self.feature_nan_count

    def get_features(self):
        return self.features

    # def list_of_pois(self, data_dict):

    #     for employee_name, attributes_dict in data_dict.items():
    #         if employee_name not in self.pois:
    #             if attributes_dict['poi']:
    #                 self.pois.append(data_dict[employee_name])
    #                 self.pois[len(self.pois)-1]['name'] = employee_name
    #                 #print('Employee {0} is POI'.format(employee_name))
    #         else:
    #             print('Info: POI already in list: ', employee_name)

    def calc_missing_data(self):
        # Calculate percentage of missing data (i.e. NaN)
        # for each feature
        df = self.enron_df
        self.feature_nan_count = {}
        for feature_name in df:
            self.feature_nan_count[feature_name] = df[feature_name].isna().sum()


    # def display_features_removed(self):
    #     pass
    
    # def filter_nans(self, percentage):
    #     pass

if __name__ == '__main__':
    dataset_file = "../dataset/final_project_dataset.pkl"
    with open(dataset_file, "br") as data_file:
        data_dict = pickle.load(data_file)

    expl = explorer(data_dict)
