import pandas
from sklearn.metrics import roc_auc_score
from numpy import exp
import math

data = pandas.read_csv('data-logistic.csv', header=None)
x1 = data[1].values[:3]
x2 = data[2].values[:3]
y = data[0].values#[:3]
l = data[0].count()
print(data)
print(x1)


a = []
w1, w2 = 0, 0
#(math.log(1 + math.frexp(-y[i]*(w1[i]*X1[i]+w2[i]*X2[i]))))
C = 0
k = 0.1
ff1, ff2 = 1, 1
f1, f2 = 0, 0
o = 0
while (ff1 or ff2) > 0.00005:
    o += 1
    sm_w1, sm_w2 = 0, 0
    for i in range(l):
        sm_w1 += y[i] * x1[i] * (1 - 1 / (1 + exp(-y[i] * (w1 * x1[i] + w2 * x2[i]))))
        sm_w2 += y[i] * x2[i] * (1 - 1 / (1 + exp(-y[i] * (w1 * x1[i] + w2 * x2[i]))))
    w1 += k * 1 / l * sm_w1 - k * C * w1
    w2 += k * 1 / l * sm_w2 - k * C * w2
    a = 1/(1 + exp(-w1*x1 - w2*x2))
    ff1 = math.sqrt((w1 - f1)**2)
    ff2 = math.sqrt((w2 - f2) ** 2)
    f1, f2 = w1, w2
print(o)
print(w1, w2)
r1 = roc_auc_score(y, a)

a = []
w1, w2 = 0, 0
#(math.log(1 + math.frexp(-y[i]*(w1[i]*X1[i]+w2[i]*X2[i]))))
C = 10
k = 0.1
ff1, ff2 = 1, 1
f1, f2 = 0, 0
o = 0
while (ff1 or ff2) > 0.00005:
    o += 1
    sm_w1, sm_w2 = 0, 0
    for i in range(l):
        sm_w1 += y[i] * x1[i] * (1 - 1 / (1 + exp(-y[i] * (w1 * x1[i] + w2 * x2[i]))))
        sm_w2 += y[i] * x2[i] * (1 - 1 / (1 + exp(-y[i] * (w1 * x1[i] + w2 * x2[i]))))
    w1 += k * 1 / l * sm_w1 - k * C * w1
    w2 += k * 1 / l * sm_w2 - k * C * w2
    a = 1 / (1 + exp(-w1 * x1 - w2 * x2))
    ff1 = math.sqrt((w1 - f1)**2)
    ff2 = math.sqrt((w2 - f2) ** 2)
    f1, f2 = w1, w2
print(o)
print(w1, w2)
r2 = roc_auc_score(y, a)
print(round(r1, 3), round(r2, 3))