import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
df = pd.read_csv('titanic.csv', index_col='PassengerId')

# Оставьте в выборке четыре признака: класс пассажира (Pclass), це-
# ну билета (Fare), возраст пассажира (Age) и его пол (Sex).
features = df.loc[:, ['Pclass', 'Fare', 'Age', 'Sex']]

# Обратите внимание, что признак Sex имеет строковые значения.
features['Sex'] = features['Sex'].map(lambda x: 1 if x == 'male' else 0)

# Выделите целевую переменную — она записана в столбце Survived.
y = df.Survived

# В данных есть пропущенные значения — например, для некоторых
# пассажиров неизвестен их возраст. Такие записи при чтении их в
# pandas принимают значение nan. Найдите все объекты, у которых
# есть пропущенные признаки, и удалите их из выборки.
features = features.dropna()
y = y[features.index.values]

# Обучите решающее дерево с параметром random_state=241 и осталь-
# ными параметрами по умолчанию.
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(features, y)

# Вычислите важности признаков и найдите два признака с наиболь-
# шей важностью. Их названия будут ответами для данной задачи
# (в качестве ответа укажите названия признаков через запятую без
# пробелов).
feature_weights = tree_classifier.feature_importances_
feature_indexes = np.argpartition(-feature_weights, 2)[:2]
print(features.columns[feature_indexes[0]] + ' ' + features.columns[feature_indexes[1]])