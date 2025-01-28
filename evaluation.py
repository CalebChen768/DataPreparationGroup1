from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def evaluate(X_train, y_train, X_test, y_test):
    clf = SVC()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return acc