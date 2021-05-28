from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import pandas as pd

"""
# Import data set
X, y = load_iris(return_X_y=True)
print("X: ", X)
print("y: ", y)



# Basic preprocessing
scaler = preprocessing.StandardScaler().fit(X)
print("sclaer.mean: ", scaler.mean_)
print("scaler.scale: ", scaler.scale_)
X_scaled = scaler.transform(X)
print("X_scaled.mean: ", X_scaled.mean(axis=0))
print("X_scaled.std: ", X_scaled.std(axis=0))
"""

"""
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

"""
# 3 users, 7 items
#X = {{"usr" : 0, "itm" : 0}, {"usr" : 0, "itm" : 3}, {"usr" : 0, "itm" : 8}, {"usr" : 1, "itm" : 2}, {"usr" : 1, "itm" : 3}, {"usr" : 2, "itm" : 4}, {"usr" : 2, "itm" : 5}}
#x = {"usr" : [0,0,0,1,1,1,2,2,2], "itm" : [10,11,12,13,14,13,14,15,14]}
d = {'col1': [1, 1, 1, 2, 2, 2], 'col2': [3, 4, 5, 6, 2, 1]}
y = [1,2,3,4,5,6]
X = pd.DataFrame(data=d)

#d = {'col1': [1, 2], 'col2': [3, 4]}
#df = pd.DataFrame(data=d)

tscv = TimeSeriesSplit(n_splits=2, test_size=2)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]