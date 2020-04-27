import numpy as np
import pandas

#data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data = pandas.read_csv('titanic.csv')
'''
answer = data['Sex'].value_counts()
#print(answer)
f = open('answer1.txt', 'w')
f.write(str(answer[0]) + ' ' + str(answer[1]))
f.close()

answer = data['Survived'].value_counts()
o = answer[1] / (answer[0] + answer[1])
answer = round((o * 100), 2)
#print(answer)
f = open('answer2.txt', 'w')
f.write(str(answer))
f.close()

db = data['Pclass'].value_counts()
f = open('answer3.txt', 'w')
o = round(db[1] / (db[1] + db[2] + db[3]) * 100, 2)
f.write(str(o))
f.close()

db = data['Age']
o1 = round(db.mean(), 2)
print(o1)
o2 = db.median()
print(o2)
f = open('answer4.txt', 'w')
f.write(str(o1) + ' ' + str(o2))
f.close()


db1 = data['SibSp']
db2 = data['Parch']
db3 = np.vstack((db1, db2))
o = np.corrcoef(db1, db2)
f = open('answer5.txt', 'w')
f.write(str(o[0][1]))
f.close()

'''
#db = data.groupby(['Name', 'Sex'])['PassengerId'].value_counts()
#db = data.groupby(['Sex'])['Name'].value_counts()['female']
db = data['Name']
f = []
f2 = []
print(db)
for row in db:
    f.append(row.split(','))
print(f)


#f = open('answer6.txt', 'w')
#f.write(str(o))
#f.close()
name_plus_sex = data['Name'] + ';' + data['Sex']

list = []

for name in name_plus_sex:

    if name.split(';')[1] == 'female':

        if '(' in name.split(';')[0]:

            splitted = name.split(';')[0].split('(')[-1].split(' ')[0]

            list.append(splitted)

        else:

            splitted_miss = name.split(';')[0].replace('Miss.','').split(',')[1]

            list.append(splitted_miss)

print(list)