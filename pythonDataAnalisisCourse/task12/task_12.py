import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

# Загрузите данные из файла abalone.csv. Это датасет, в котором требуется предсказать возраст
# ракушки (число колец) по физическим измерениям.
df = pd.read_csv('abalone.csv')

# Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
#  Если вы используете Pandas, то подойдет следующий код:
#  data[’Sex’] = data[’Sex’].map(lambda x: 1 if x == ’M’ else (-1 if x == ’F’ else 0))
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# Разделите содержимое файлов на признаки и целевую переменную.
# В последнем столбце записана целевая переменная, в остальных — признаки.
X = df.drop('Rings', 1)
y = df['Rings']

# Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев:
# от 1 до 50 (random_state=1). Для каждого из вариантов оцените качество работы полученного леса
# на кросс-валидации по 5 блокам. Используйте параметры "random_state=1"и "shuffle=True"
# при создании генератора кросс-валидации sklearn.cross_validation.KFold В качестве меры качества
# воспользуйтесь долей правильных ответов (sklearn.metrics.r2_score).
scores = {}
for i in range(1, 50):
    forest = RandomForestRegressor(random_state=1, n_estimators=i)
    forest.fit(X, y)
    cv = KFold(random_state=1, shuffle=True, n_splits=5)
    scores[i] = cross_val_score(forest, X, y, cv=cv, scoring='r2').mean()
print(scores)

# Определите, при каком минимальном количестве деревьев случайный лес показывает качество на
# кросс-валидации выше 0.52. Это количество и будет ответом на задание.
for score in scores.items():
    if score[1] > 0.52:
        trees_quantity = score[0]
    break
print(trees_quantity)
# Обратите внимание на изменение качества по мере роста числа деревьев. Ухудшается ли оно?
