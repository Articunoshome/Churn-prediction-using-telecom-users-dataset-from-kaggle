# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:51:27 2021

@author: Aryaan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def draw_confusion_matrix(y_true, y_pred, classes=None, normalize=True, title=None, cmap=plt.cm.Blues):
    acc = np.sum(y_true == y_pred) / len(y_true)
    print(('\x1b[32m' + 'Accuracy = {}' + '\x1b[0m').format(acc))

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Confusion Matrix = \n{}'.format(np.round(cm, 3)))

    if classes is None:
        classes = [str(i) for i in range(len(np.unique(y_true)))]

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True',
           xlabel='Predicted')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


df = pd.read_csv('telecom_users.csv')

df=df.drop(['Unnamed: 0','customerID'],axis=1)
df['MonthlyCharges'] = df['MonthlyCharges'].astype(int)
df['TotalCharges'] = df['TotalCharges'].astype(str)
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
df['TotalCharges'] = df['TotalCharges'].astype(float)
total_charge = df['TotalCharges']
df['TotalCharges'] = df['TotalCharges'].astype(int)



to_predict = df['Churn']
df=df.drop('Churn',axis=1)

catcol=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService', 'OnlineSecurity',	
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod']

df = pd.get_dummies(df, columns=catcol)
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(df, to_predict, random_state=24)
print("Shape of features_train :", X_train.shape)
print("Shape of features_test :", X_test.shape)
print("Shape of target_train :", y_train.shape)
print("Shape of target_test :", y_test.shape)

scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def do_pca_and_plot_roc(clf):
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)
    clf.fit(X_t_train, y_train)
    metrics.plot_roc_curve(clf, X_t_test, y_test)
    plt.show()

def do_pca_and_plot_decision_regions(clf):
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)
    clf.fit(X_t_train, y_train)
    plot_decision_regions(X_t_train, y_train.to_numpy(), clf=clf, legend=2)

model = SVC()
model.fit(X_train, y_train)

predict = model.predict(X_test)
print(metrics.f1_score(y_test, predict, average='weighted'))
draw_confusion_matrix(y_test, predict, title="model")

do_pca_and_plot_roc(SVC())


