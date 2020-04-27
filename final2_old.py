import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
import time
import datetime


#   загружаем данные, отрезаем ненужные столбцы: номер матча, время проведения, тип лобби
data = pd.read_csv('features.csv').head(100).iloc[:, 3:]

#   выделяем матрицу целевой переменной
y = data['radiant_win']

#   удаляем ненужные столбцы
data = data.drop(['duration',
           'tower_status_radiant',
           'tower_status_dire',
           'barracks_status_radiant',
           'barracks_status_dire',
           'radiant_win',]
          , axis=1)
data1 = data
#   заполняем пустые значения в матрице признаков
#data1 = data1.fillna(0)

#scale = sklearn.preprocessing.StandardScaler()
#n_data1 = scale.fit_transform(data1)

#   формируем генератор для кросс-валидации
#KFold = KFold(shuffle=True, n_splits=5)

'''
#   ищу параметр C
parametrs = {'C':[1, 10]}
clf = LogisticRegression(random_state=241, solver='lbfgs')
gs = GridSearchCV(clf, parametrs, scoring='roc_auc', cv=KFold)
gs.fit(n_data1, y)
print(clf.C)


#   проверил топорным методом

for C in range(1, 10):
    clf = LogisticRegression(solver='lbfgs', C=C, random_state=241)
    clf.fit(n_data1, y)
    cvs = cross_val_score(clf, n_data1, y, scoring='roc_auc', cv=KFold)
    print(np.mean(cvs))
'''
#n_clf = LogisticRegression(solver='lbfgs', C=1, random_state=241)
#n_clf.fit(n_data, y)

data2 = data.drop(['r1_hero',
                  'r2_hero',
                  'r3_hero',
                  'r4_hero',
                  'r5_hero',
                  'd1_hero',
                  'd2_hero',
                  'd3_hero',
                  'd4_hero',
                  'd5_hero']
          , axis=1)
#n_data2 = scale.fit_transform(data)
'''
#   ищу параметр C
parametrs = {'C':[1, 10]}
clf = LogisticRegression(random_state=241, solver='lbfgs')
gs = GridSearchCV(clf, parametrs, scoring='roc_auc', cv=KFold)
gs.fit(n_data2, y)
print(clf.C)


#   проверил топорным методом

for C in range(1, 10):
    clf = LogisticRegression(solver='lbfgs', C=C, random_state=241)
    clf.fit(n_data2, y)
    cvs = cross_val_score(clf, n_data2, y, scoring='roc_auc', cv=KFold)
    print(np.mean(cvs))
'''
#n_clf = LogisticRegression(solver='lbfgs', C=10, random_state=241)
#n_clf.fit(n_data, y)

#   определяю количество разных героев
data_dop = data.get(['r1_hero',
                  'r2_hero',
                  'r3_hero',
                  'r4_hero',
                  'r5_hero',
                  'd1_hero',
                  'd2_hero',
                  'd3_hero',
                  'd4_hero',
                  'd5_hero'])
#print(data_dop.nunique(axis=0))

# N — количество различных героев в выборке
X_pick = np.zeros((data.shape[0], 112))

for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

data3 = data2.join(data_dop)


def clf_func(data):

    #   заполняю матрицу пустыми значениями
    data = data.fillna(0)

    #   привожу переменные к одному масштабу
    scale = sklearn.preprocessing.StandardScaler()
    n_data = scale.fit_transform(data)

    #   формируем генератор для кросс-валидации
    KFold = sklearn.model_selection.KFold(shuffle=True, n_splits=5)

    #   ищу параметр C
    parametrs = {'C':[1, 10]}
    clf = LogisticRegression(random_state=241, solver='lbfgs')
    gs = GridSearchCV(clf, parametrs, scoring='roc_auc', cv=KFold)
    gs.fit(n_data, y)
    print(clf.C)


    #   проверил топорным методом

    for C in range(1, 11):
        clf = LogisticRegression(solver='lbfgs', C=C, random_state=241)
        clf.fit(n_data, y)
        cvs = cross_val_score(clf, n_data, y, scoring='roc_auc', cv=KFold)
        print(C, np.mean(cvs))
    return print('Done')

print('data1')
clf_func(data1)
print('data2')
clf_func(data2)
print('data3')
clf_func(data3)