import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
scaler = StandardScaler()

#data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data1 = pandas.read_csv('perceptron-train.csv ')
data2 = pandas.read_csv('perceptron-test.csv')
X = np.array(data1.iloc[:, 1:].values)
y = np.array(data1.iloc[:, 0].values)
clf = Perceptron(random_state=241)
clf.fit(X, y)
predictions = clf.predict(X)
X2 = np.array(data2.iloc[:, 1:].values)
y2 = np.array(data2.iloc[:, 0].values)
predictions = clf.predict(X2)
ac = accuracy_score(y2, predictions)
print(ac)
### scaled
data1 = pandas.read_csv('perceptron-train.csv ', header=None)
data2 = pandas.read_csv('perceptron-test.csv', header=None)
X = np.array(data1.iloc[:, 1:].values)
X = scaler.fit_transform(X)
y = np.array(data1.iloc[:, 0].values)
clf = Perceptron()
clf.fit(X, y)
predictions = clf.predict(X)
X2 = np.array(data2.iloc[:, 1:].values)
X2 = scaler.transform(X2)
y2 = np.array(data2.iloc[:, 0].values)
predictions = clf.predict(X2)
ac2 = accuracy_score(y2, predictions)

print(ac2)
answer = round(ac2 - ac, 3)
print(answer)

f = open('answer.txt', 'w')
f.write(str(answer))
f.close()

'''
X = data.loc[:, x_labels]
X = X.dropna()
X['Sex'] = X['Sex'].map(lambda sex: 1 if sex == 'male' else 0)

y = data['Survived']
print(y)
y = y[X.index.values]
print(y)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(np.array(X.values), np.array(y.values))


importances = pandas.Series(clf.feature_importances_, index=x_labels)
answer = (importances.sort_values(ascending=False).head(2).index.values)



'''