"""
Chapter 3 - Exercise 3: Titanic

Tackle the Titanic dataset. A great place to start is on Kaggle
(https://kaggle.com/c/titanic). Alternatively download the data from
https://homl.info/titanic.tgz and unzip the tarball like you did for the
housing data in Chapter 2.

This gives you two CSV files, train.csv and test.csv, which you load with
pandas.read_csv(). The goal is to train a classifier that can predict the
'Survived' column based on the other columns.

Approach / notes
- Load train.csv and inspect with .head(), .info(), .describe()
- Handle missing values (Age has lots, Cabin has many, Embarked has a couple)
- Encode categorical columns (Sex, Embarked) with OneHotEncoder or OrdinalEncoder
- Try a few classifiers: RandomForestClassifier, SVC, LogisticRegression
- Use cross validation to pick the best
"""

from dataclasses import field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


# --- Solution ---
# Start writing your solution here.
# 1) load the data
train_data = pd.read_csv("classification/train.csv")
test_data = pd.read_csv("classification/test.csv")

#2) inspect, clean and encode the data
#print(train_data.head().to_string())
#print(train_data.info()) # Expected: 889 samples. Age: 714, Cabin: 204, Embarked: 889
#print(train_data.describe())
#print(train_data.shape)

#Next steps:
#Remove Cabin column (204), Remove additional Embarked rows (2), Reomve Age values (177) rows
train_data_clean = train_data.drop(columns = ["Cabin"])
#print(train_data_clean[train_data_clean['Embarked'].isna()].to_string())
train_data_clean = train_data_clean.dropna(subset=["Embarked"])
median_age = train_data_clean['Age'].median()
train_data_clean["Age"] = train_data_clean["Age"].fillna(median_age)

train_data_clean["Sex"] = train_data_clean["Sex"].map({"male":0, "female": 1})
train_data_clean = pd.get_dummies(train_data_clean, columns=["Embarked"])
train_data_clean["Embarked_C"] = train_data_clean["Embarked_C"].map({False:0, True: 1})
train_data_clean["Embarked_Q"] = train_data_clean["Embarked_Q"].map({False:0, True: 1})
train_data_clean["Embarked_S"] = train_data_clean["Embarked_S"].map({False:0, True: 1})

train_data_clean = train_data_clean.drop(columns=["PassengerId", "Name", "Ticket"])
print (train_data_clean.head().to_string())