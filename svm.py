"""
Created on Fri Aug 8 03:27:24 2018

@author: Rajendra Prasad
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime
start=(datetime.now().minute*60*1000+datetime.now().second*1000+datetime.now().microsecond*0.001)
sns.set(color_codes=True)
dataset=pd.read_csv('GL.csv')
dataset.head()
print(dataset.head())
dataset = dataset.drop('ID',axis=1)
dataset.head()
print(dataset.head())
print(dataset.shape)
print(dataset.describe())
print(dataset.groupby('status').size())
dataset.plot(kind='box', sharex=False, sharey=False)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
end = datetime.now().minute*60*1000+datetime.now().second*1000+datetime.now().microsecond*0.001
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('accuracy is',accuracy_score(y_pred,y_test))
print("Execution time: %.2f ms" % (abs(end-start)))
i=0
for i in range(0,49):
    print(classifier.predict([X_test[i]]))

