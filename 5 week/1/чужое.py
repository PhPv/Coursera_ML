import pandas
import sklearn
import scipy.sparse
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor


data = pandas.read_csv('abalone.csv').head(1)
sex = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
#X = data.iloc[:, 1:-1]
X = scipy.sparse.hstack([sex, data.iloc[:, 1:-1]])
y_t = data['Rings'].values

KFold = sklearn.model_selection.KFold(random_state=1, shuffle=True, n_splits=5)




for F in range(1, 5):
    y_pred = []
    forest = RandomForestRegressor(random_state=1, n_estimators=F)

    for train_index, test_index in KFold.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y_t[train_index], y_t[test_index]

        forest.fit(X_train, y_train)
        cv_pred = forest.predict(X_test)

        y_pred[test_index] = cv_pred
        metrics = sklearn.metrics.r2_score(y_test, cv_pred)
        print(metrics)
    #cvs = sklearn.model_selection.cross_val_score(forest, X, cv=KFold, scoring='r2')
    #metrics = sklearn.metrics.r2_score(y_t, y_p)
    #print(metrics)