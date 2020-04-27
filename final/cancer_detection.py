import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import time
import datetime

#   градиентный бустинг
print('градиентный бустинг')
#   загружаем данные, отрезаем ненужные столбцы: номер матча, время проведения, тип лобби
data = pd.read_csv('features.csv').iloc[:, 3:]

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



#   записываем в файл информацию по числу записей в каждом столбце
#f = data.count()
#f.to_csv('answer1')

#   заполняем пустые значения в матрице признаков
data = data.fillna(0)

#   формируем генератор для кросс-валидации
KFold = KFold(shuffle=True, n_splits=5)

#   создаем цикл для оценки качества кросс-валидации для разного количества деревьев
for n_estimators in [10, 20, 30]:

#   оценка затрачиваемого времени
    start_time = datetime.datetime.now()
    time.sleep(3)

#   созданием и обучение модели
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(data, y)

#   оценка качества
    cvs = cross_val_score(clf, data, y, scoring='roc_auc', cv=KFold)
    print(np.mean(cvs), 'Time elapsed:', datetime.datetime.now() - start_time)


# логистическая регрессия
print('логистическая регрессия')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#   загружаем данные, отрезаем ненужные столбцы: номер матча, время проведения, тип лобби
data = pd.read_csv('features.csv')
data_test = pd.read_csv('features_test.csv')

#   выделяем матрицу целевой переменной
y = data['radiant_win']

#   удаляем ненужные столбцы, не относящиеся к первым 5 минутам
data = data.drop(['duration',
           'tower_status_radiant',
           'tower_status_dire',
           'barracks_status_radiant',
           'barracks_status_dire',
           'radiant_win',

           'start_time']
          , axis=1)


#   заполняю пропущенные значения
data = data.fillna(0)
data_test = data_test.fillna(0)

data_test = data_test.drop([

           'start_time']
          , axis=1)

#   форумируем первый семпл данных
data1 = data.iloc[:, 1:]
data1_test = data_test.iloc[:, 1:]

#   формирую второй семпл данных без категориальных признаков
data2 = data.drop(['r1_hero',
                  'r2_hero',
                  'r3_hero',
                  'r4_hero',
                  'r5_hero',
                  'd1_hero',
                  'd2_hero',
                  'd3_hero',
                  'd4_hero',
                  'd5_hero',
                  'lobby_type']
          , axis=1).iloc[:, 1:]

data2_test = data_test.drop(['r1_hero',
                  'r2_hero',
                  'r3_hero',
                  'r4_hero',
                  'r5_hero',
                  'd1_hero',
                  'd2_hero',
                  'd3_hero',
                  'd4_hero',
                  'd5_hero',
                  'lobby_type']
          , axis=1).iloc[:, 1:]


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

data_test_dop = data_test.get(['r1_hero',
                  'r2_hero',
                  'r3_hero',
                  'r4_hero',
                  'r5_hero',
                  'd1_hero',
                  'd2_hero',
                  'd3_hero',
                  'd4_hero',
                  'd5_hero'])


#   получаю ответ на вопрос
#print(data_dop.nunique(axis=0))

# N — количество различных героев в выборке
X_pick = np.zeros((data.shape[0], 112))


for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_pick_test = np.zeros((data_test.shape[0], 112))
for i, match_id in enumerate(data_test.index):
    for p in range(5):
        X_pick[i, data_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1


#   формирую 3ий семпл данных
data3 = np.column_stack((data2, X_pick))
data3_test = np.column_stack((data2_test, X_pick_test))



#   функция с обучающей моделью
def clf_func(data):
    #data = np.delete(data, 0, 1)
    #   заполняю матрицу пустыми значениями
    #data = data.fillna(0)

    #   привожу переменные к одному масштабу
    scale = sklearn.preprocessing.StandardScaler()
    n_data = scale.fit_transform(data)

    #   формируем генератор для кросс-валидации
    KFold = sklearn.model_selection.KFold(shuffle=True, n_splits=5)

    #   ищу параметр C
    parametrs = {'C': np.power(10.0, np.arange(-5, 6))}
    clf = LogisticRegression(random_state=241, solver='lbfgs')
    gs = GridSearchCV(clf, parametrs, scoring='roc_auc', cv=KFold)
    gs.fit(n_data, y)
    print(gs.best_params_['C'])
    print(gs.best_score_)
    return gs.best_params_['C']

#прогон обучающей модели для первого  (полного) семпла данных
clf_func(data1)

#прогон обучающей модели для второго (без категориальных признаков) семпла данных
clf_func(data2)

#прогон обучающей модели для третьего семпла данных
C_opt = clf_func(data3)



#C_opt = 0.01
#n_data3 = data3.fillna(0)
#n_data3_test = data3_test.fillna(0)
scale = sklearn.preprocessing.StandardScaler()
n_data3 = scale.fit_transform(data3)
n_data3_test = scale.fit_transform(data3)
n_clf = LogisticRegression(random_state=241, C=C_opt, solver='lbfgs')
n_clf.fit(n_data3, y)
n_clf.predict(n_data3_test)
pred = n_clf.predict_proba(n_data3_test)[:, 1]
print(pred)
print(max(pred))
print(min(pred))

