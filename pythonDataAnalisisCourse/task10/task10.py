from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from scipy.sparse import hstack
import re

# Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла
# salary-train.csv.
df = pd.read_csv('salary-train.csv')


# Проведите предобработку:
# • Приведите тексты к нижнему регистру.
# • Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова.
def text_prepare(row):
    row['FullDescription'] = row['FullDescription'].lower()
    row['FullDescription'] = re.sub('\W', ' ', row['FullDescription'])
    return row


df['FullDescription'] = df.apply(text_prepare, axis=1)

# • Примените TfidfVectorizer для преобразования текстов в векторы признаков.
# Оставьте только те слова, которые встречаются хотя бы в 5 объектах
# (параметр min_df у TfidfVectorizer).
tf_idf_vect = TfidfVectorizer(min_df=5)
X_text = tf_idf_vect.fit_transform(df.FullDescription)

# • Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку ’nan’.
df['LocationNormalized'].fillna('nan', inplace=True)
df['ContractTime'].fillna('nan', inplace=True)

# • Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized
# и ContractTime.
enc = DictVectorizer()
X_categ = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))

# • Объедините все полученные признаки в одну матрицу "объекты-признаки".
# Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
# Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
X = hstack([X_text, X_categ])
y = df['SalaryNormalized']

# Обучите гребневую регрессию с параметром alpha=1. Целевая переменная записана в столбце
# SalaryNormalized.
model = Ridge(alpha=1)
model.fit(X, y)

# Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел
test = pd.read_csv('salary-test-mini.csv')
test['FullDescription'] = test.apply(text_prepare, axis=1)
X_test_text = tf_idf_vect.transform(test.FullDescription)
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_categ])

y_test = model.predict(X_test)
print(y_test)