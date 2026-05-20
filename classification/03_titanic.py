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
titanic_data = pd.read_csv("classification/train.csv")

#2) inspect, clean and encode the data
#print(titanic_data.head().to_string())
#print(titanic_data.info()) # Expected: 889 samples. Age: 714, Cabin: 204, Embarked: 889
#print(titanic_data.describe())
#print(titanic_data.shape)

#Next steps:
#Remove Cabin column (204), Remove additional Embarked rows (2), Reomve Age values (177) rows
titanic_data_clean = titanic_data.drop(columns = ["Cabin"])
#print(titanic_data_clean[titanic_data_clean['Embarked'].isna()].to_string())
titanic_data_clean = titanic_data_clean.dropna(subset=["Embarked"])
median_age = titanic_data_clean['Age'].median()
titanic_data_clean["Age"] = titanic_data_clean["Age"].fillna(median_age)

titanic_data_clean["Sex"] = titanic_data_clean["Sex"].map({"male":0, "female": 1})
titanic_data_clean = pd.get_dummies(titanic_data_clean, columns=["Embarked"])
titanic_data_clean["Embarked_C"] = titanic_data_clean["Embarked_C"].map({False:0, True: 1})
titanic_data_clean["Embarked_Q"] = titanic_data_clean["Embarked_Q"].map({False:0, True: 1})
titanic_data_clean["Embarked_S"] = titanic_data_clean["Embarked_S"].map({False:0, True: 1})
titanic_data_clean = titanic_data_clean.drop(columns=["PassengerId", "Name", "Ticket"])

print (titanic_data_clean.head().to_string())

#3) split into X and y and train test split

train_y = titanic_data_clean["Survived"]
train_x = titanic_data_clean.drop(columns=["Survived"])
test_x = pd.read_csv("classification/test.csv")





