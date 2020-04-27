import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors.classification import KNeighborsClassifier
scaler = StandardScaler()

data = pandas.read_csv('wine.data', header=None)
#dat = data[11]
#data = data[0:10]

data1 = np.array(data.iloc[:, 0].values)
data2 = np.array(data.iloc[:, 1:].values)
data2 = scale(data2)
#dat1 = np.array(dat.iloc[:, 0].values)
m = 5
gen = KFold(shuffle=True, n_splits=m, random_state=42)
reso = []
index = []
resmax = 0
for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    res = cross_val_score(neigh, data2, data1, cv=gen)
    res = np.mean(res)
    print(res)
    if res > resmax:
        resmax = res
        index = k
print(index, resmax)
