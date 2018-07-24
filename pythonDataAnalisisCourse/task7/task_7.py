from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV

# Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям
# "космос"и "атеизм"(инструкция приведена выше).
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

# Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем
# вам вычислить TF-IDF по всем данным. При таком подходе получается, что признаки на обучающем
# множестве используют информацию из тестовой выборки — но такая ситуация вполне законна,
# поскольку мы не используем значения целевой переменной из теста. На практике нередко
# встречаются ситуации, когда признаки объектов тестовой выборки известны на момент обучения,
# и поэтому можно ими пользоваться при обучении алгоритма.
X = newsgroups.data
y = newsgroups.target
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Подберите минимальный лучший параметр C из множества [10 −5 , 10 −4 , ...10 4 , 10 5 ]
# для SVM с линейным ядром (kernel=’linear’) при помощи кросс валидации по 5 блокам.
# Укажите параметр random_state=241 и для SVM, и для KFold. В качестве меры качества используйте
# долю верных ответов (accuracy).
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
model = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(model, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
best_c = gs.best_params_.get('C')
# best_c = 1.0

# Обучите SVM по всей выборке с лучшим параметром C, найденным на предыдущем шаге.
model = SVC(kernel='linear', C=best_c, random_state=241)
model.fit(X, y)

# Найдите 10 слов с наибольшим по модулю весом. Они являются ответом на это задание.
# Укажите их через запятую, в нижнем регистре, в лексикографическом порядке.
words = vectorizer.get_feature_names()
coef_index = sorted(zip(model.coef_.data, model.coef_.indices),
                    key=lambda x: abs(x[0]), reverse=True)
coef_index = list(coef_index)
top_words = [words[coef_index[i][1]] for i in range(10)]
print(sorted(top_words))