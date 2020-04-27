import pandas
import sklearn
from sklearn import feature_extraction
import scipy.sparse
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.decomposition import PCA

prices = pandas.read_csv('close_prices.csv').iloc[:, 1:]
index = pandas.read_csv('djia_index.csv').iloc[:, 1:]

PCA1 = PCA(n_components=1)
PCA1.fit(prices)
o = PCA1.transform(prices)
print(o)
#print(PCA.explained_variance_ratio_)
#print(sum(PCA.explained_variance_ratio_))
PCA2 = PCA(n_components=10)
PCA2.fit(prices)
o2 = PCA2.transform(prices)

print(np.corrcoef(o.T, index.T))
m = max(PCA.components_)
#print(m)
k = 0
for x in PCA.components_[0]:
    if x != m:
        k += 1
    else:
        print(k)

prices = pandas.read_csv('close_prices.csv', header=None).head(1)
#print(prices)
#print(prices[+1])



#np.corrcoef()




'''
FullDescription = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True).values
FullDescription_test = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True).values

LocationNormalized = data['LocationNormalized'].fillna('nan', inplace=True)#.values
ContractTime = data['ContractTime'].fillna('nan', inplace=True)#.values
SalaryNormalized = data['SalaryNormalized'].values

TFD = feature_extraction.text.TfidfVectorizer(min_df=5)

for x in range(len(FullDescription)):
    FullDescription[x] = FullDescription[x].lower()
for x in range(len(FullDescription_test)):
    FullDescription_test[x] = FullDescription_test[x].lower()

TFD_FullDescription = TFD.fit_transform(FullDescription)
TFD_FullDescription_test = TFD.transform(FullDescription_test)

DV = feature_extraction.DictVectorizer()
data_categ = DV.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
test_categ = DV.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))


Xtrain = scipy.sparse.hstack([TFD_FullDescription, data_categ])
Xtest = scipy.sparse.hstack([TFD_FullDescription_test, test_categ])
#new_data = scipy.sparse.hstack(TFD_FullDescription, data_categ)



R = Ridge(alpha=1, random_state=241) #fit_intercept=False, solver='lsqr')
R.fit(Xtrain, SalaryNormalized)
print(np.round(R.predict(Xtest), 2))
'''