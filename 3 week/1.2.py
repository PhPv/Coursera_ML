
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
X = newsgroups.data
y = newsgroups.target
print(type(X))

TFIDF = feature_extraction.text.TfidfVectorizer()
X = TFIDF.fit_transform(X)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
#print(clf.C)
print(gs.best_params_)
print(gs.cv_results_['param_C'])

clf1 = svm.SVC(kernel='linear', random_state=241, C=1)
clf1.fit(X, y)
cf = clf1.coef_

#sm = abs(csc_matrix.todense(cf))
q = pd.DataFrame(cf.toarray()).transpose()
top10 = abs(q).sort_values([0], ascending=False).head(10).index
ff = TFIDF.get_feature_names()

aa = []
for n in top10:
    aa.append(ff[n])
aa.sort()
f = open('1.2 answer.txt', 'w')
for x in aa:
    f.write(str(x) + ' ')
f.close()


