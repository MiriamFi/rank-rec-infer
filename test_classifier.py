from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
print("X: ", X)
print("y: ", y)
clf = LogisticRegression(random_state=0, max_iter=120).fit(X, y)
print(clf.predict(X[:2, :]), "\n\n")
print(clf.predict_proba(X[:2, :]), "\n\n")
print(clf.score(X, y))