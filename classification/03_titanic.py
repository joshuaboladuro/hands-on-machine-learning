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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


# --- Solution ---
# Start writing your solution here.

if __name__ == "__main__":
    pass
