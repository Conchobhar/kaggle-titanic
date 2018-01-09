'''
Outdated
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import random as rnd
import csv
import matplotlib.pyplot as plt

titanic_dir_train = "/home/xu/Work/kaggle/titanic/data/train.csv" 
titanic_dir_test  = "/home/xu/Work/kaggle/titanic/data/test.csv"

data_train = pd.read_csv(titanic_dir_train)
data_test  = pd.read_csv(titanic_dir_test)

################################################################################
# Data Exploration
################################################################################
# See here for more:
# http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
def data_exploration(data_train):
    print(data_train.columns.values)
    print(data_train.head())
    data_train.info()
    print('_'*40)
    data_train.describe()

    # Pivoting data to compare
    data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

################################################################################
# Feature Engineering:
# Taken from https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic
################################################################################

# NaN padding:
from sklearn.preprocessing import Imputer

def nan_padding(data, columns):
    for column in columns:
        imputer=Imputer(missing_values = 'NaN', strategy = 'mean')
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data

nan_columns = ["Age", "SibSp", "Parch"]

data_train = nan_padding(data_train, nan_columns)
data_test = nan_padding(data_test, nan_columns)

# pandas.get_dummies - Convert categorical variable into dummy/indicator variables:
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["Pclass"]
data_train = dummy_data(data_train, dummy_columns)
data_test = dummy_data(data_test, dummy_columns)

# Encode 'Sex' as ints (1 and 0)
from sklearn.preprocessing import LabelEncoder

def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male","female"])
    data["Sex"]=le.transform(data["Sex"]) 
    return data

data_train = sex_to_int(data_train)
data_test = sex_to_int(data_test)

# Normalize data
from sklearn.preprocessing import MinMaxScaler

def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data
data_train = normalize_age(data_train)
data_test = normalize_age(data_test)

# Remove columns if necessary
def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

data_test_passenger_id = data_test["PassengerId"]
not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]
data_train = drop_not_concerned(data_train, not_concerned_columns)
data_test = drop_not_concerned(data_test, not_concerned_columns)

# Binarize and split set
# Neither are used here yet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def split_valid_test_data(data, fraction = (1 - 0.8)):
    data_y = pd.DataFrame({"Survived" : data["Survived"]})
    #lb = LabelBinarizer()
    #data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction, random_state = 7)

    return train_x, train_y, valid_x, valid_y 

train_x, train_y, valid_x, valid_y = split_valid_test_data(data_train, fraction = (1 - 0.8))
full_train_x, full_train_y, _, _ = split_valid_test_data(data_train, fraction = 0)

data_train.head()