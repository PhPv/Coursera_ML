import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import time
import datetime


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
for n_estimators in [10, 20, 30, 50, 100]:

#   оценка затрачиваемого времени
    start_time = datetime.datetime.now()
    time.sleep(3)

#   созданием и обучение модели
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(data, y)

#   оценка качества
    cvs = cross_val_score(clf, data, y, scoring='roc_auc', cv=KFold)
    print(np.mean(cvs), 'Time elapsed:', datetime.datetime.now() - start_time)
