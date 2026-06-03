 """
Chapter 3 - Exercise 4: Spam Classifier

Build a spam classifier (a more challenging exercise).

a. Download examples of spam and ham from Apache SpamAssassin's public
   datasets (https://homl.info/spamassassin).
b. Unzip the datasets and familiarise yourself with the data format.
c. Split the data into a training set and a test set.
d. Write a data preparation pipeline to convert each email into a feature
   vector. Each email becomes a (sparse) vector indicating which words are
   present (e.g. [1, 0, 0, 1, ...] means word #1 and word #4 are present).
e. Train several classifiers and pick the one with high precision AND high
   recall on the test set.

Approach / notes
- Download via urllib.request or tarfile
- Use Python's email module to parse messages
- Build a custom BaseEstimator/TransformerMixin pipeline step for preprocessing
- Vectorise with CountVectorizer (or build a sparse matrix manually)
- Try LogisticRegression, MultinomialNB, RandomForestClassifier
- Report both precision_score and recall_score (and F1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


# --- Solution ---
# Start writing your solution here.

if __name__ == "__main__":
    pass
