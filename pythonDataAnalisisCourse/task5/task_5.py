import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.
train = pd.read_csv('perceptron-train.csv')
test = pd.read_csv('perceptron-test.csv')

X_train = train[train.columns[1:train.shape[1]]]
y_train = train[train.columns[0]]

X_test = test[test.columns[1:test.shape[1]]]
y_test = test[test.columns[0]]

# Обучите персептрон со стандартными параметрами и random_state=241.
perceptron = Perceptron(random_state=241)
perceptron.fit(X_train, y_train)

# Подсчитайте качество (долю правильно классифицированных объ-
# ектов, accuracy) полученного классификатора на тестовой выборке.
predicts = perceptron.predict(X_test)
acc = accuracy_score(y_test, predicts)

# Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
# Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.fit_transform(X_test)
perceptron.fit(train_scaled, y_train)
predicts_after_scaled = perceptron.predict(test_scaled)
acc_after_scaled = accuracy_score(y_test, predicts_after_scaled)

# Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
#  Это число и будет ответом на задание.
print(acc_after_scaled - acc)