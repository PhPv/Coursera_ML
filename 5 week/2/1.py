import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn import ensemble
from math import exp
import numpy
import matplotlib.pyplot as plt

data = pandas.read_csv('gbm-data.csv')
y = data['Activity'].values
X = data.iloc[:, 1:].values
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.8, random_state=241)



x = [0.2] #[1, 0.5, 0.3, 0.2, 0.1]
for el in x:
    f = []
    GBS = ensemble.GradientBoostingClassifier(n_estimators=250,
                                              verbose=True,
                                              random_state=241,
                                              learning_rate=el)
    GBS.fit(X_train, y_train)
    o = GBS.staged_decision_function(X_test)
    for i, y_pred in enumerate(o):
        y_pred = 1 / (1 + numpy.exp(-y_pred))
        ll = log_loss(y_test, y_pred)
        f.append(ll)

    plt.figure()
    plt.plot(f, 'r', linewidth=2)
    plt.show()

clf = RandomForestClassifier(random_state=241, n_estimators=36)
clf.fit(X_train, y_train)
ll2 = log_loss(y_test, clf.predict_proba(X_test))
print(ll2)
