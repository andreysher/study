import pandas as pd
from sklearn.svm import SVC

# Загрузите выборку из файла svm-data.csv. В нем записана двумерная выборка
# (целевая переменная указана в первом столбце, признаки — во втором и третьем).
df = pd.read_csv('svm-data.csv')
y = df[df.columns[0]]
X = df[df.columns[1:]]

# Обучите классификатор с линейным ядром, параметром C = 100000 и random_state=241.
# Такое значение параметра нужно использовать, чтобы убедиться,что SVM работает с выборкой как
# с линейноразделимой. При более низких значениях параметра алгоритм будет настраиваться с учетом
# слагаемого в функционале, штрафующего за маленькие отступы, из-за чего результат может не совпасть
# с решением классической задачи SVM для линейно разделимой выборки.
svc = SVC(kernel='linear', C=100000, random_state=241)
svc.fit(X, y)

# Найдите номера объектов, которые являются опорными (нумерация с единицы).
# Они будут являться ответом на задание. Обратите внимание, что в качестве ответа нужно привести
# номера объектов в возрастающем порядке через запятую. Нумерация начинается с 1.
so = svc.support_
so.sort()
print(' '.join([str(i+1) for i in so]))