from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии
# торгов за каждый день периода.
df = pd.read_csv('close_prices.csv')
X = df.drop('date', 1)

# На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
pca = PCA(n_components=10)
pca.fit(X)

# Скольких компонент хватит, чтобы объяснить 90% дисперсии?
disp = 0
component_quantity = 0
for score in pca.explained_variance_ratio_:
    component_quantity += 1
    disp += score
    if disp >= 0.9:
        break
# print(component_quantity)

# Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
components = pd.DataFrame(pca.transform(X))
compotent1 = components[0]

# Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
dji = pd.read_csv('djia_index.csv')

# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
corr = np.corrcoef(compotent1, dji['^DJI'])
# print(corr)

# Какая компания имеет наибольший вес в первой компоненте?
company_weights = sorted(pca.components_[0], key=lambda x: abs(x), reverse=True)
top_company = df.columns[pca.components_[0].tolist().index(company_weights[0])+1]
print(top_company)