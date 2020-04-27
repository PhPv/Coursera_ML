import pandas as pd

import numpy as np

from sklearn import datasets

newsgroups=datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer()

X=vectorizer.fit_transform(newsgroups.data)

y=newsgroups.target

from sklearn.svm import SVC

from sklearn.model_selection import KFold

kf=KFold(n_splits=5, shuffle=True, random_state=241)

from sklearn.model_selection import GridSearchCV

grid={'C':np.power(10.0,np.arange(-5,6))}

clf=SVC(kernel='linear',random_state=241)

gs=GridSearchCV(clf,grid, scoring='accuracy',cv=kf)


clf2=SVC(kernel='linear',C=1.0, random_state=241)

clf2.fit(X,y)

coef=clf2.coef_
#print(coef)
q=pd.DataFrame(coef.toarray()).transpose()
top10=abs(q).sort_values([0], ascending=False).head(10)
top101=abs(q).sort_values([1], ascending=False).head(10)
print(top10)
print(top101)
indices=[]

indices=top10.index

words=[]

for i in indices:

    feature_mapping=vectorizer.get_feature_names()

    words.append(feature_mapping[i])

print(sorted(words))
