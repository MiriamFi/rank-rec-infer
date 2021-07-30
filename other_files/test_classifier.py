from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

import pandas as pd


# Import data set
#X, y = load_iris(return_X_y=True)
#print("X: ", X)
#print("y: ", y)



#print("y: ", y)
"""
# Basic preprocessing
scaler = preprocessing.StandardScaler().fit(X)
print("sclaer.mean: ", scaler.mean_)
print("scaler.scale: ", scaler.scale_)
X_scaled = scaler.transform(X)
print("X_scaled.mean: ", X_scaled.mean(axis=0))
print("X_scaled.std: ", X_scaled.std(axis=0))



#Basic logistic regression
clf = LogisticRegression(random_state=0, max_iter=120).fit(X, y)
print(clf.predict(X[:2, :]), "\n\n")
print(clf.predict_proba(X[:2, :]), "\n\n")
print(clf.score(X, y))


#X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
print(pipe.fit(X_train, y_train) ) # apply scaling on training data
print(pipe.predict(X_test), "\n\n")
print(pipe.predict_proba(X_test), "\n\n")
print(pipe.score(X_test, y_test))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), svm.SVC(probability=True))
print(pipe.fit(X_train, y_train) ) # apply scaling on training data
print(pipe.predict(X_test), "\n\n")
print(pipe.predict_proba(X_test), "\n\n")
print(pipe.score(X_test, y_test))


clf = LogisticRegression(solver="liblinear").fit(X,y)
print(roc_auc_score(y, clf.predict_proba(X), multi_class='ovr'))
"""

import numpy as np
from sklearn.dummy import DummyClassifier
X = np.array([-1, 1, 1, 1])
y = np.array([0, 1, 1, 2])
y = y.reshape(-1, 1)

enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()
print(y)
dummy_clf = DummyClassifier(strategy="most_frequent")
print(dummy_clf.fit(X, y))
print(dummy_clf.predict(X))
print(dummy_clf.score(X, y))