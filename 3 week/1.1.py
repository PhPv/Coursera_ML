import pandas
from sklearn import svm



data = pandas.read_csv('svm-data.csv', header=None)
y = data.iloc[:, 0]
x = data.iloc[:, 1:]

svc = svm.SVC(kernel='linear', random_state=241, C=100000)
svc.fit(x, y)
print(svc.support_)

f = open('1.1 answer.txt', 'w')
for x in svc.support_:
    f.write(str(x + 1) + ' ')
f.close()