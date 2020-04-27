import pandas
import sklearn
from sklearn import feature_extraction
import scipy.sparse
from sklearn.linear_model import Ridge
import numpy as np


data = pandas.read_csv('salary-train.csv')
test = pandas.read_csv('salary-test-mini.csv')

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
