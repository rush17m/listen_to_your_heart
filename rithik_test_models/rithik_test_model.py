import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt


np.set_printoptions(precision=4)
seg_length = 1500

x = None
y = None

for db in ["chf2db", "chfdb"]:
    for record in os.listdir(f'data/{db}/rr'):
        rr = np.loadtxt(f'data/{db}/rr/{record}')
        n_segments = rr.shape[0] // seg_length
        rr = rr[:n_segments*seg_length].reshape((n_segments, seg_length))
        labels = np.c_[np.ones(n_segments), np.zeros(n_segments)]

        if x is None or y is None:
            x = rr
            y = labels
        else:
            x = np.r_[rr, x]
            y = np.r_[labels, y]


for db in ["nsrdb", "nsr2db", "fantasia"]:
    for record in os.listdir(f'data/{db}/rr'):
        rr = np.loadtxt(f'data/{db}/rr/{record}')
        n_segments = rr.shape[0] // seg_length
        rr = rr[:n_segments*seg_length].reshape((n_segments, seg_length))
        labels = np.c_[np.zeros(n_segments), np.ones(n_segments)]

        if x is None or y is None:
            x = rr
            y = labels
        else:
            x = np.r_[rr, x]
            y = np.r_[labels, y]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x.shape, y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def decisionTreeClassifier(X_train, X_test, y_train, y_test):
    print('DECISION TREE CLASSIFIER')
    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, predictions))


def randomForestClassifier(X_train, X_test, y_train, y_test):
    print('RANDOM FOREST CLASSIFIER')
    clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can experiment with the number of estimators

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, predictions))


def kNeighborsClassifier(X_train, X_test, y_train, y_test):
    print('KNeighbors CLASSIFIER')
    clf = KNeighborsClassifier()

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, predictions))


def xgBoostClassifier(X_train, X_test, y_train, y_test):
    print('XGBOOST REGRESSION')
    clf = XGBClassifier()

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, predictions))


# # INCOMPLETE
# def svmClassifier(X_train, X_test, y_train, y_test):
#     print('SUPPORT VECTOR MACHINE CLASSIFIER')
#     clf = SVC(kernel='linear', C=1.0)  # You can experiment with different kernels and C values

#     clf.fit(X_train, y_train)

#     predictions = clf.predict(X_test)

#     accuracy = accuracy_score(y_test, predictions)
#     print(f'Accuracy: {accuracy}')
#     print('Classification Report:')
#     print(classification_report(y_test, predictions))


# # INCOMPLETE
# def naiveBayesGausianClassifier(X_train, X_test, y_train, y_test):
#     print('NAIVE BAYES CLASSIFIER')
#     clf = GaussianNB()

#     clf.fit(X_train, y_train)

#     predictions = clf.predict(X_test)

#     accuracy = accuracy_score(y_test, predictions)
#     print(f'Accuracy: {accuracy}')
#     print('Classification Report:')
#     print(classification_report(y_test, predictions))


# # INCOMPLETE
# def logisticRegression(X_train, X_test, y_train, y_test):
#     print('LOGISTIC REGRESSION')
#     clf = LogisticRegression()

#     clf.fit(X_train, y_train)

#     predictions = clf.predict(X_test)

#     accuracy = accuracy_score(y_test, predictions)
#     print(f'Accuracy: {accuracy}')
#     print('Classification Report:')
#     print(classification_report(y_test, predictions))



decisionTreeClassifier(X_train, X_test, y_train, y_test)
randomForestClassifier(X_train, X_test, y_train, y_test)
kNeighborsClassifier(X_train, X_test, y_train, y_test)
xgBoostClassifier(X_train, X_test, y_train, y_test)

# Still figuring these out
# logisticRegression(X_train, X_test, y_train, y_test)
# svmClassifier(X_train, X_test, y_train, y_test)
# naiveBayesGausianClassifier(X_train, X_test, y_train, y_test)
