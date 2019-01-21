from sklearn import datasets
df = datasets.load_iris()

import pandas as pd

x_train = pd.DataFrame(df.data)
x_train.columns = df.feature_names
y_train = pd.DataFrame(df.target)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
clf.score(x_train, y_train)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, clf.predict(x_train))

