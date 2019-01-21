from sklearn import datasets
from sklearn.linear_model import LogisticRegression

cancer_db = datasets.load_breast_cancer()

clf = LogisticRegression()
clf.fit(cancer_db.data, cancer_db.target)

clf.score(cancer_db.data, cancer_db.target)

clf.predict_proba(cancer_db.data)

clf.predict(cancer_db.data) - cancer_db.target