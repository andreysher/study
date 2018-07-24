import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

# Загрузите выборку wine.data
df = pd.read_csv('wine.data')

# Извлеките из данных признаки и классы. Класс записан в первом
# столбце (три варианта), признаки — в столбцах со второго по по-
# следний. Более подробно о сути признаков можно прочитать по
# адресу https://archive.ics.uci.edu/ml/datasets/Wine
classes = df[df.columns[0]]
features = df[df.columns[1:len(df.columns)]]

# Оценку качества необходимо провести методом кросс-валидации по
# 5 блокам (5-fold). Создайте генератор разбиений, который переме-
# шивает выборку перед формированием блоков (shuffle=True). Для
# воспроизводимости результата, создавайте генератор KFold с фик-
# сированным параметром random_state=42. В качестве меры каче-
# ства используйте долю верных ответов (accuracy).
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

# Найдите точность классификации на кросс-валидации для метода
# k ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при
# k от 1 до 50. При каком k получилось оптимальное качество? Чему
# оно равно (число в интервале от 0 до 1)? Данные результаты и
# будут ответами на вопросы 1 и 2.
print('Before scale')
def getAccuracy(kfolds, features, classes):
    scores = []
    for k in range(1,51):
        model = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(model, features, classes, cv=kfolds, scoring='accuracy').mean())
    return scores

scores = getAccuracy(kfolds,features,classes)
top_score = max(scores)
print('Top accuracy = ' + str(top_score))
best_k = scores.index(top_score) + 1
print('Best K = ' + str(best_k))


# Произведите масштабирование признаков с помощью функции sklearn.preprocessing.sc
features = scale(features)

# Снова найдите оптимальное k на кросс-валидации.
print('After scale')
def getAccuracy(kfolds, features, classes):
    scores = []
    for k in range(1,51):
        model = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(model, features, classes, cv=kfolds, scoring='accuracy').mean())
    return scores

scores = getAccuracy(kfolds,features,classes)
top_score = max(scores)
print('Top accuracy = ' + str(top_score))
best_k = scores.index(top_score) + 1
print('Best K = ' + str(best_k))

# Какое значение k получилось оптимальным после приведения при-
# знаков к одному масштабу? Как изменилось значение качества?
# Приведите ответы на вопросы 3 и 4.


