import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("tools/")
#sys.path.append("dataset/")
sys.path.append("data_wrangling/")

#print(sys.path)

from data_explorer import explorer
from feature_format import featureFormat, targetFeatureSplit

from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] 

### Load the dictionary containing the dataset
data_dict = {}

dataset_file = "dataset/final_project_dataset.pkl"
with open(dataset_file, "br") as data_file:
    data_dict = pickle.load(data_file)

def repeat_to_length(string_to_expand, length):
    return (string_to_expand * (int(length/len(string_to_expand))+1))[:length]

# for k, v in data_dict.items():
#     l = 45 - len(k)
#     dots = repeat_to_length('.', l)
#     print('\n\t{0}: {1} {2}'.format(k, dots, v))

#--- Explore dataset
eobj = explorer(data_dict)
parameters = eobj.get_dataset_parameters()
enron_df = eobj.get_dataframe()

#--- Show estatistics:
print('\nDataset statistical information:')
print('================================')
enron_df.info()

# #--- List people in dataset:
# for k in data_dict.keys():
#     print(k)

#--- Show an example of dataset contents for POI Kenneth Lay:
print('\nExample of data content for POI Kenneth Lay:')
print('============================================')
sample_contents = data_dict['LAY KENNETH L']
for k, v in sample_contents.items():
    l = 45 - len(k)
    dots = repeat_to_length('.', l)
    print('\t{0}: {1} {2}'.format(k, dots, v))

def convert_to_csv(data_dict):
    #--- Save dataset as CSV

    # Create ouput directory if it doesn't already exist:
    outdirname = 'output'
    try:
        os.makedirs('output')
        print('\nDirectory "{}" created'.format(outdirname))
    except FileExistsError:
        print('\nDirectory "{}" already exists - nothing done.'.format(outdirname))

    csvfilename = outdirname + '/enron_data.csv'

    with open(csvfilename, 'w') as csv_file:
        print('\nWriting data set to ../{}'.format(csvfilename))
        writer = csv.writer(csv_file)

        #--- create header:
        header_line = ['Name']

        for name in data_dict.keys():
            if name != 'TOTAL':
                value_pairs = data_dict[name]

                for feature_value in value_pairs.keys():
                    header_line.append(feature_value)

                break

        writer.writerow(header_line)

        for name in data_dict.keys():
            line = []
            line.append(name)

            for k, v in data_dict[name].items():
                line.append(v)

            writer.writerow(line)

print('\nWrite dataset to csv file:')
print('================================================')
convert_to_csv(data_dict)

print('\nSummary of information contained in the dataset:')
print('================================================')
dataset_params = eobj.get_dataset_parameters()

for k,v in dataset_params.items():
    if k != 'Features with NaN':
        l = 45 - len(k)
        dots = repeat_to_length('.', l)
        print('\t{0}: {1} {2}'.format(k, dots, v))


#--- Calculate ratio of NaN entries in the dataset for each feature:
print('\nRatio of NaN entries per feature in dataset:')
print('============================================')
features_nan_counts = eobj.get_feature_nan_counts()
df_features_nan_counts = pd.DataFrame(features_nan_counts.items())
df_features_nan_counts.columns = ['Feature','NaN count']
df_features_nan_counts.sort_values(by=['NaN count'], inplace=True)
df_features_nan_counts.reset_index(drop=True, inplace=True)

max_nan_count = df_features_nan_counts.iloc[len(df_features_nan_counts)-1]['NaN count']
print('\nMax NaN count: {}'.format(max_nan_count))
df_features_nan_counts['ratio'] = df_features_nan_counts['NaN count']/max_nan_count
print(df_features_nan_counts.to_string(index=False))

#--- list features with more than selected percentage (%) NaNs:
thresh_ratio = 0.7
rslt_df = df_features_nan_counts.loc[df_features_nan_counts['ratio'] > thresh_ratio]
nan_features = rslt_df['Feature'].tolist()
print('\nFeatures with NaN > {} % will be excluded:'.format(thresh_ratio*100.0))
print(nan_features)

selected_features = []
all_features = eobj.get_features()

for feature in all_features:
    if feature not in nan_features:
        selected_features.append(feature)

print('\nselected features:\n{}'.format(selected_features))


#--- Calculate ratio of NaN entries in the dataset for each employee:
print('\nNaN entries per employee in dataset:')
print('============================================')
df_employee_nan_counts = ((enron_df[all_features].isna()).sum(axis=1)).to_frame()
df_employee_nan_counts['Name'] = df_employee_nan_counts.index
df_employee_nan_counts.columns = ['NaN count', 'Name']
df_employee_nan_counts = df_employee_nan_counts[['Name', 'NaN count']]
df_employee_nan_counts['NaN %'] = (df_employee_nan_counts['NaN count'] / len(all_features))*100.
df_employee_nan_counts.sort_values(by=['NaN count'], inplace=True)
df_employee_nan_counts.reset_index(drop=True, inplace=True)
print(df_employee_nan_counts)

empl_to_pop = df_employee_nan_counts.loc[df_employee_nan_counts['NaN %'] >90. , 'Name'].tolist()

size_empl_to_pop = len(empl_to_pop)

if size_empl_to_pop > 1:
    print('\nThe following employees have more than 90% NaN entries and will be ignored if non-POI:')
    print(empl_to_pop)
elif size_empl_to_pop == 1:
    print('\nThe following employee has more than 90% NaN entries and will be ignored if non-POI:')
    print(empl_to_pop[0])
else:
    pass

poi_list = eobj.get_poi_list()

for empl in empl_to_pop:
    if empl not in poi_list:
        data_dict.pop(empl, 0 )


### Task 2: Remove outliers

#--- plot points
plot_features = ["poi", "salary", "bonus"]


def show_data(title, features):
    ### read in data dictionary, convert to numpy array
    data = featureFormat(data_dict, [features[1], features[2], 'poi'])

    for point in data:
        salary = point[0]
        bonus = point[1]
        poi = point[2]

        if poi:
            color = 'red'
        else:
            color = 'green'

        plt.scatter( salary, bonus, color=color)

    plt.title(title)
    plt.xlabel(features[1].capitalize() )
    plt.ylabel(features[2].capitalize() )
    plt.show()

# show_data('All points', plot_features)

#--- remove 'TOTAL' point:
data_dict.pop( 'TOTAL', 0 ) 

# show_data('"TOTAL" entry removed', plot_features)

def create_df_from_csv(filename):
    return pd.read_csv(filename)


#--- create dataframe from updated dictionary dataset:
convert_to_csv(data_dict)
df_enron_reloaded = create_df_from_csv('output/enron_data.csv')

print(df_enron_reloaded)

#--- further explore the new dataset by looking into each feature:

#--- Money related features:
money_features = ['salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 
    'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock']
#--- Email related features:
email_features = ['email_address', 'from_poi_to_this_person','from_this_person_to_poi','from_messages',
    'shared_receipt_with_poi','to_messages']

#--- Explore money features:
#--- Compare POI data vs non-POI data:

print('\nComparisson of non-POI vs POI salaries:')
print('=======================================:')

#--- Extract pois salaries:
pois_salaries = pd.DataFrame([], columns=['Name','Salary'])
pois_salaries['Name'] = df_enron_reloaded.loc[df_enron_reloaded['poi']==True]['Name']
pois_salaries['Salary'] = df_enron_reloaded.loc[df_enron_reloaded['poi']==True]['salary']

#--- Compute IQR:
Q1 = pois_salaries['Salary'].quantile(0.25)
Q3 = pois_salaries['Salary'].quantile(0.75)
poi_IQR = Q3 - Q1

print('\nPOI salaries:\n', pois_salaries)
print('\nPOI IQR=', poi_IQR)

#--- Extract non-poi salaries:
non_pois_salaries = pd.DataFrame([], columns=['Name','Salary'])
non_pois_salaries['Name'] = df_enron_reloaded.loc[df_enron_reloaded['poi']==False]['Name']
non_pois_salaries['Salary'] = df_enron_reloaded.loc[df_enron_reloaded['poi']==False]['salary']

#--- Compute IQR:
Q1 = non_pois_salaries['Salary'].quantile(0.25)
Q3 = non_pois_salaries['Salary'].quantile(0.75)
non_poi_IQR = Q3 - Q1
print('\nNon-POI salaries:\n', non_pois_salaries)
print('\nNon-POI IQR=', non_poi_IQR)

#--- Plot IQRs:
import seaborn as sns

# sns.boxplot(x='poi', y='salary', data=df_enron_reloaded)
# df_enron_reloaded.boxplot(column=['salary'])




#--- Compare poi vs non-poi:

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# my_dataset = data_dict

### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# dump_classifier_and_data(clf, my_dataset, features_list)
