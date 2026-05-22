import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

np.random.seed(42)

# 1) Load the data
train_titanic = pd.read_csv("classification/train.csv").dropna(subset=["Embarked"])
test_titanic  = pd.read_csv("classification/test.csv")

#2) Inspect, clean and encode the data
#Remove Cabin column rows (204), Remove additional Embarked rows (2), Fill Age values (177) with median
def clean(df, median_age):
    df = df.drop(columns=["Cabin", "Name", "Ticket"])
    df["Age"] = df["Age"].fillna(median_age)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], dtype=int)
    return df

median_age = train_titanic["Age"].median()
train_clean = clean(train_titanic, median_age).drop(columns=["PassengerId"])
test_ids    = test_titanic["PassengerId"]
test_clean  = clean(test_titanic, median_age).drop(columns=["PassengerId"])

#3) Evaluate the models, train the best one and make predictions for the test features
train_y = train_clean["Survived"]
train_x = train_clean.drop(columns=["Survived"])

log_clf = LogisticRegression(max_iter=1000)
rf_clf  = RandomForestClassifier(random_state=42)
svc_clf = SVC()

for name, clf in [("Logistic Regression", log_clf), ("Random Forest", rf_clf), ("SVM", svc_clf)]:
    score = cross_val_score(clf, train_x, train_y, cv=5, scoring="accuracy")
    print(f"{name} accuracy: {score.mean():.2f}")
'''
Logistic Regression accuracy: 0.79
Random Forest accuracy: 0.82 (Winner)
SVM accuracy: 0.67
'''

rf_clf.fit(train_x, train_y)
preds = rf_clf.predict(test_clean)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
submission.to_csv("classification/submission.csv", index=False)
print("Submission file created successfully.")


