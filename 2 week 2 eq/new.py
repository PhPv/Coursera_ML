import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston


data = load_boston()
data1 = np.array(data['data'])
data1sc = scale(data1)
data2 = np.array(data['target'])
data2 = data2.reshape(-1, 1)

reso = []
index = []
resmax = -1000000
for n in np.linspace(1, 10, 200, endpoint=True):
    gen = KFold(shuffle=True, n_splits=5, random_state=42)
    neighR = KNeighborsRegressor(metric='minkowski', n_neighbors=5, weights='distance', p=n)
    res = cross_val_score(neighR, data1sc, data2, scoring='neg_mean_squared_error', cv=gen)
    res = np.mean(res)
    if res > resmax:
        resmax = res
        index = n
print(index, resmax)
