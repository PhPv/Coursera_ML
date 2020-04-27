import pandas
import sklearn
import scipy.sparse
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = pandas.read_csv('abalone.csv')
sex = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
XX = data.iloc[:, 1:-1]
X = np.column_stack((sex, XX))
#X = scipy.sparse.vstack([sex, XX])
#X = scipy.sparse.hstack([sex, data.iloc[:, 1:-1]])
y_t = data['Rings'].values



KFold = sklearn.model_selection.KFold(random_state=1, shuffle=True, n_splits=5)
fff = 0
for F in range(1, 51):
    forest = RandomForestRegressor(random_state=1, n_estimators=F)
    forest.fit(X, y_t)
    y_p = forest.predict(X)

    cvs = sklearn.model_selection.cross_val_score(forest, X, y_t, cv=KFold, scoring='r2')
    metrics = round(np.mean(cvs), 2)
    fff += 1
    print(fff, metrics)
