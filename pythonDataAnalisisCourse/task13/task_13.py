import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy
# (параметр values у датафрейма). В первой колонке файла с данными записано, была или нет реакция.
# Все остальные колонки (d1 - d1776) содержат различные характеристики молекулы,
# такие как размер, форма и т.д. Разбейте выборку на обучающую и тестовую,
# используя функцию train_test_split с параметрами test_size = 0.8 и random_state = 241.
df = pd.read_csv('gbm-data.csv')
X = df.drop('Activity', 1).values
y = df['Activity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


# Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True,
# random_state=241 и для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1]
# проделайте следующее:
# • Используйте метод staged_decision_function для предсказания качества на обучающей и тестовой
# выборке на каждой итерации.
# • Преобразуйте полученное предсказание по формуле сигмоиды где y_pred — предсказаное значение.
def sigmoid(y_pred):
    return 1.0 / (1.0 + math.exp(-y_pred))


def get_log_loss(model, X, y):
    scores = []
    for pred in model.staged_decision_function(X):
        predictions = [sigmoid(y_pred) for y_pred in pred]
        loss = log_loss(y, predictions)
        scores.append(loss)
    return scores


def plot_log_loss(train_loss, test_loss, learning_rate):
    plt.figure()
    plt.plot(test_loss, 'r')
    plt.plot(train_loss, 'b')
    plt.legend(['test', 'train'])
    plt.savefig('rate_' + str(learning_rate) + '.png')


learningRate_minLossVal_Iter = {}
for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    gr_boost = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
    gr_boost.fit(X_train, y_train)
    train_loss = get_log_loss(gr_boost, X_train, y_train)
    test_loss = get_log_loss(gr_boost, X_test, y_test)
    plot_log_loss(train_loss, test_loss, lr)
    learningRate_minLossVal_Iter[lr] = (min(test_loss), test_loss.index(min(test_loss)))
# • Вычислите и постройте график значений log-loss на обучаю-
# щей и тестовой выборках, а также найдите минимальное зна-
# чение метрики и номер итерации, на которой оно достигается.


# 3. Как ведет себя график качества на тестовой выборке с уменьшени-
# ем параметра learning_rate? Обратите внимание, что чем меньше
# learning_rate, тем позднее алгоритм начинаем переобучаться.

# Приведите минимальное значение log-loss на тестовой выборке и номер итерации,
# при котором оно достигается, при learning_rate = 0.2
min_loss_val = learningRate_minLossVal_Iter[0.2][0]
min_loss_iter = learningRate_minLossVal_Iter[0.2][1]
print(min_loss_val)
print(min_loss_iter)

# 4. На этих же данных обучите RandomForestClassifier с количеством
# деревьев, равным количеству итераций, на котором достигается
# наилучшее качество у градиентного бустинга из предыдущего пунк-
# та, random_state=241 и остальными параметрами по умолчанию.
# Какое значение log-loss на тесте получается у этого случайного леса? (Не забывайте, что
# предсказания нужно получать с помощью функции predict_proba)
random_forest = RandomForestClassifier(n_estimators=min_loss_iter, random_state=241)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict_proba(X_test)
test_log_loss = log_loss(y_test, predictions)
print(test_log_loss)