import pandas as pd
import math
from sklearn.metrics import roc_auc_score

# Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная на которой
# принимает значения -1 или 1.
df = pd.read_csv('data-logistic.csv', header=None)
y = df[0]
X = df.loc[:, 1:]


# Убедитесь, что выше выписаны правильные формулы для градиентного спуска.
def stepW1(w1, w2, y, X, k, c):
    l = len(y)
    s = 0
    for i in range(l):
        s += y[i] * X[1][i] * (1.0 - (1.0 / (1.0 + math.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i])))))
    return w1 + (k / l) * s - k * c * w1


def stepW2(w1, w2, y, X, k, c):
    l = len(y)
    s = 0
    for i in range(l):
        s += y[i] * X[2][i] * (1.0 - (1.0 / (1.0 + math.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i])))))
    return w2 + (k / l) * s - k * c * w2


# Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10)
# логистической регрессии. Используйте длину шага k=0.1.
def grad_descent(X, y, c=0.0, w1=0.0, w2=0.0, k=0.1, error=1e-5, max_iter=10000):
    i = 1
    w1_opt = stepW1(w1, w2, y, X, k, c)
    w2_opt = stepW2(w1, w2, y, X, k, c)
    err = math.sqrt(math.pow((w1_opt - w1), 2) + math.pow((w2_opt - w2), 2))

    while err > error and i < max_iter:
        i += 1
        w1, w2 = w1_opt, w2_opt
        w1_opt = stepW1(w1, w2, y, X, k, c)
        w2_opt = stepW2(w1, w2, y, X, k, c)
        err = math.sqrt(math.pow((w1_opt - w1), 2) + math.pow((w2_opt - w2), 2))

    return w1_opt, w2_opt


# Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов
# на соседних итерациях должно быть не больше 1e-5). Рекомендуется ограничить сверху число
# итераций десятью тысячами.
w1, w2 = grad_descent(X, y)
regw1, regw2 = grad_descent(X, y, c=10.0)


# Какое значение принимает AUC-ROC на обучении в случае с регуляризацией и без нее?
# Эти величины будут ответом на задание
def get_prob(x, w1, w2):
    return 1.0 / (1.0 + math.exp(-x[1] * w1 - x[2] * w2))


prob = X.apply(lambda x: get_prob(x, w1, w2), axis=1)
reg_prob = X.apply(lambda x: get_prob(x, regw1, regw2), axis=1)

auc = roc_auc_score(y, prob)
reg_auc = roc_auc_score(y, reg_prob)
print(str(auc) + ' ' + str(reg_auc))
