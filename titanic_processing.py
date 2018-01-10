'''
Implementing:
https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html

Incl. approaching the full data set as global variable that each function changes,
rather than passing the data to each function as an argument.
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import random as rnd
import csv
import matplotlib.pyplot as plt

def get_combined_data():
    titanic_dir_train = "./data/train.csv" 
    titanic_dir_test  = "./data/test.csv"
    
    train = pd.read_csv(titanic_dir_train)
    test  = pd.read_csv(titanic_dir_test)
    test_passenger_id = test["PassengerId"]

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    
    return combined, test_passenger_id, targets

combined, test_passenger_id, targets = get_combined_data()

################################################################################
# Feature Engineering:
# Taken from https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic
################################################################################

def get_titles():

    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)

get_titles()

# Fill in nan values, grouping by some column (in this case "Title")
def numerical_nan_groupby_fill():

    global combined

    nan_columns = ["Age", "SibSp", "Parch", "Fare"]
    groupby_col = "Title"

    for col in nan_columns:
        combined[col] = combined.groupby(groupby_col)[col].transform(lambda x: x.fillna(x.median()))        

numerical_nan_groupby_fill()

def categorical_nan_fill():

    global combined

    combined.Fare.fillna("S", inplace = True)

categorical_nan_fill()
## sklearn mean approach:
# from sklearn.preprocessing import Imputer

# def nan_padding(data, columns):
#     for column in columns:
#         imputer=Imputer(missing_values = 'NaN', strategy = 'mean')
#         data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
#     return data

# nan_columns = ["Age", "SibSp", "Parch"]

# combined = nan_padding(combined, nan_columns)

# pandas.get_dummies - Convert categorical variable into dummy/indicator variables
# and drop original column:
def process_cabin():
    
    global combined
    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

process_cabin()
        
def generate_dummy_data():
    
    global combined

    dummy_columns = ["Pclass", "Title", "Embarked", "Cabin"]
    
    for column in dummy_columns:
        combined = pd.concat([combined, pd.get_dummies(combined[column], prefix=column)], axis=1)
        #combined = combined.drop(column, axis=1) # This is now done in "drop_not_concerned"

generate_dummy_data()

# Encode 'Sex' as ints (1 and 0)

def sex_to_int():
    
    from sklearn.preprocessing import LabelEncoder

    global combined

    le = LabelEncoder()
    le.fit(["male","female"])
    combined["Sex"] = le.transform(combined["Sex"]) 

sex_to_int()

def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

process_ticket()

# Normalize data

def normalize_age():

    from sklearn.preprocessing import MinMaxScaler
    
    global combined

    scaler = MinMaxScaler()
    combined["Age"] = scaler.fit_transform(combined["Age"].values.reshape(-1,1))

normalize_age()

# Remove columns if necessary
def drop_not_concerned():

    global combined

    drop_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked", "Pclass", "Title"]
    
    combined.drop(drop_columns, axis = 1, inplace = True)

drop_not_concerned()

def recover_train_test_target():

    global combined

    train = combined.head(891)
    test = combined.iloc[891:]
    
    return train, test

train, test = recover_train_test_target()

train_with_targets = pd.concat([train, targets], axis = 1)

# Binarize and split set
# Neither are used here yet
def split_valid_test_data(data, fraction = (1 - 0.8)):
    
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    
    data_y = pd.DataFrame({"Survived" : data["Survived"]})
    
    #lb = LabelBinarizer()
    #data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction, random_state = 7)

    return train_x, train_y, valid_x, valid_y 

train_x, train_y, valid_x, valid_y = split_valid_test_data(train_with_targets, fraction = (1 - 0.8))
full_train_x, full_train_y, _, _ = split_valid_test_data(train_with_targets, fraction = 0)

train.head()