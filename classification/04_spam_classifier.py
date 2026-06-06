"""
Chapter 3 - Exercise 4: Spam Classifier

Build a spam classifier (a more challenging exercise).


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
from pathlib import Path 

np.random.seed(42)


# --- Solution ---
# Start writing your solution here.

if __name__ == "__main__":
    pass

from pathlib import Path 

spam_dir = [
    p for p in Path("classification/spam").iterdir()
    if "Zone.Identifier" not in p.name
    ]
ham_dir = [
    p for p in Path("classification/easy_ham").iterdir()
    if "Zone.Identifier" not in p.name
    ]

spam_count = len(spam_dir)
ham_count = len(ham_dir)

#print(f"Spam count: {spam_count}")
#print(f"Ham count: {ham_count}")

#split into train and test sets (80% train, 20% test)
spam_train = spam_dir[:int(0.8 * spam_count)]
ham_train = ham_dir[:int(0.8 * ham_count)]
spam_test = spam_dir[int(0.8 * spam_count):]
ham_test = ham_dir[int(0.8 * ham_count):]


# convert emails to feature vectors 
def read_email(file):
    with open(file, "r", encoding="latin-1") as f:
        return f.read()

#print(spam_dir[100])
#print(read_email(spam_dir[100]))  
print(spam_dir[0])
print(read_email(spam_dir[1]))

