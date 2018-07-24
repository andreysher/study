import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект, у которого признаки записаны в поле data,
# а целевой вектор — в поле target.
boston = load_boston()
features = boston.data
target = boston.target

# Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
features = scale(features)


# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом,
# чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).
# Используйте KNeighborsRegressor с n_neighbors=5 и
# weights='distance' — данный параметр добавляет в алгоритм веса,
# зависящие от расстояния до ближайших соседей.
# В качестве метрики качества используйте среднеквадратичную ошибку
# (параметр scoring='mean_squared_error' у cross_val_score;
#  при использовании библиотеки scikit-learn версии 0.18.1 и выше необходимо указывать
# scoring='neg_mean_squared_error'). Качество оценивайте, как и в предыдущем задании,
#  с помощью кросс-валидации по 5 блокам с random_state = 42,
#  не забудьте включить перемешивание выборки (shuffle=True).
def getMSE(kfolds, features, target):
    scores = []
    p_range = np.linspace(1, 10, 200)
    for p in p_range:
        knn_reg = KNeighborsRegressor(p=p, n_neighbors=5, weights='distance')
        scores.append(cross_val_score(knn_reg, features, target, \
                                      cv=kfolds, scoring='neg_mean_squared_error').mean())
    return scores, p_range


kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
mses, p_values = getMSE(kfolds, features, target)
print(mses)

# Определите, при каком p качество на кросс-валидации оказалось оптимальным.
# Обратите внимание, что cross_val_score возвращает массив показателей качества по блокам;
# необходимо максимизировать среднее этих показателей.
# Это значение параметра и будет ответом на задачу.
best_mse = max(mses)
print('Best mse = ' + str(best_mse))
best_p = p_values[mses.index(best_mse)]
print('Best p = ' + str(best_p))
