import pandas as pd
from sklearn import metrics
# Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true)
# и ответы некоторого классификатора (колонка predicted).
df = pd.read_csv('classification.csv')

# Заполните таблицу ошибок классификации:
# Для этого подсчитайте величины TP, FP, FN и TN согласно иx определениям.
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1.
# Ответ в данном вопросе — четыре числа через пробел.
true_positive = df.loc[(df.true == 1) & (df.pred == 1)].shape[0]
true_negative = df.loc[(df.true == 0) & (df.pred == 0)].shape[0]
false_positive = df.loc[(df.true == 0) & (df.pred == 1)].shape[0]
false_negative = df.loc[(df.true == 1) & (df.pred == 0)].shape[0]

print(str(true_positive) + ' ' + str(false_positive) +
      ' ' + str(false_negative) + ' ' + str(true_negative))

# Посчитайте основные метрики качества классификатора:
some_metrics = {}
some_metrics['accuracy'] = metrics.accuracy_score(df.true, df.pred)
some_metrics['precision'] = metrics.precision_score(df.true, df.pred)
some_metrics['recall'] = metrics.recall_score(df.true, df.pred)
some_metrics['f1_score'] = metrics.f1_score(df.true, df.pred)
print(some_metrics)

# Посчитайте площадь под ROC-кривой для каждого классификатора.
# Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца
#  с ответами этого классификатора)? Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
clf_data = pd.read_csv('scores.csv')
scores = {}
for classifier in clf_data.columns[1:]:
    scores[classifier] = metrics.roc_auc_score(clf_data.true, clf_data[classifier])
scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print(scores)

# Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70%
# (укажите название столбца с ответами этого классификатора)?
# Какое значение точности при этом получается?
precision_recall_scores = {}
for classifier in clf_data.columns[1:]:
    precision, recall, thresholds = metrics.precision_recall_curve(clf_data.true, clf_data[classifier])
    pr_df = pd.DataFrame({'precision': precision, 'recall': recall})
    precision_recall_scores[classifier] = pr_df.loc[pr_df.recall >= 0.7]['precision'].max()
precision_recall_scores = sorted(precision_recall_scores.items(), key=lambda x: x[1], reverse=True)
print(precision_recall_scores)