import pandas
import sklearn.metrics

data = pandas.read_csv('classification.csv')
'''
T = data['true'].values
P = data['pred'].values
l = data['true'].count()
TP, FP, FN, TN = 0, 0, 0, 0
y = 0
for i in range(l):
    if T[i] == 1:
        if P[i] == 1:
            TP += 1
        else:
            FN += 1
        y += 1
    else:
        if P[i] == 1:
            FP += 1
        else:
            TN += 1

acc = sklearn.metrics.accuracy_score(T, P)
pre = sklearn.metrics.precision_score(T, P)
rec = sklearn.metrics.recall_score(T, P)
f = sklearn.metrics.f1_score(T, P)
'''
data = pandas.read_csv('scores.csv')

logreg = data['score_logreg'].values
svm = data['score_svm'].values
knn = data['score_knn'].values
tree = data['score_tree'].values
y = data['true'].values
sklearn.metrics.roc_auc_score(y, svm)
f = [logreg, svm, knn, tree]
all = []
mall = []
for n in range(4):
    roc = sklearn.metrics.roc_auc_score(y, f[n])
    rc = sklearn.metrics.precision_recall_curve(y, f[n])
    for x in range(len(rc[0])):
        if rc[1][x] >= 0.7:
            all.append((rc[0][x]))
    print(max(all))
    mall.append(max(all))
print(mall)