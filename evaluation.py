from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def evaluate(X_train, y_train, X_test, y_test):

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    # scores = cross_val_score(SVC(kernel='linear'), X_train, y_train, cv=5)
    return acc