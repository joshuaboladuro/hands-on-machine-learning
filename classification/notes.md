# Chapter 3 – Classification

My working notes for the four end-of-chapter exercises in Hands-On Machine Learning by Aurélien Géron.

## Exercises

1. **MNIST classifier (97%+ accuracy)** – build a classifier for MNIST that beats 97% on the test set. Hint: KNeighborsClassifier, grid search over `weights` and `n_neighbors`.
2. **MNIST data augmentation** – shift each training image by one pixel in each of the four directions, retrain the best model on the expanded set, compare accuracy.
3. **Titanic** – Kaggle Titanic dataset, predict the `Survived` column.
4. **Spam classifier** – Apache SpamAssassin data. Build a preprocessing pipeline (headers, lowercase, punctuation, URL/NUMBER tokens, optional stemming), turn each email into a sparse word vector, try several classifiers, optimise for both high precision and high recall.

## Concepts to lock in

- Binary vs multiclass vs multilabel vs multioutput classification
- Confusion matrix, precision, recall, F1
- Precision/recall tradeoff and the `decision_function` threshold
- ROC curve, AUC, when to use ROC vs PR curves
- Cross validation (`cross_val_score`, `cross_val_predict`)
- OvR (One vs Rest) vs OvO (One vs One) for multiclass

## Gotchas / things to remember

- _Add notes here as you hit them._

## Useful imports

```python
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## References (don't commit these)

- Géron's solution notebook: https://homl.info/colab3
