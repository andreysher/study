import pandas as pd
import re

df = pd.read_csv('titanic.csv', index_col='PassengerId')

answers = open('answers_1.txt', 'w')

# какое количество мужчин и женщин было на корабле
counts_sex = df['Sex'].value_counts()
print(str(counts_sex['female']) + ' ' + str(counts_sex['male']), file=answers)

# какой части пасажиров удалось выжить
number_of_passengers = df.shape[0]
print(df.loc[df.Survived == 1].shape[0] / number_of_passengers, file=answers)

# какую долю пасажиры первого класса составляли среди всех пасажиров
print(df.loc[df.Pclass == 1].shape[0] / number_of_passengers, file=answers)

# среднее и медиана возраста пасажиров
# чтобы понять как попытка восстановить возраст повлияла на среднее и медиану сделаю без налов
know_age = df.dropna(subset=['Age'], how='all')
print(str(know_age['Age'].mean()) + ' ' + str(know_age['Age'].median()), file=answers)

# коррелирует ли число братьев/сестер с числом родителей/детей
print(df['SibSp'].cov(df['Parch']) / df.SibSp.var() * df.Parch.var(), file=answers)


# какое самое популярное женское имя на корабле
def getName(name):
    res = re.search('^[^,]+, (.*)', name)
    if res:
        name = res.group(1)

    res = re.search('\(([^)]+)\)', name)
    if res:
        name = res.group(1)

    name = name.replace('"', '')
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)
    name = name.rsplit(' ', 1)[0]
    return name


girlsNames = df[df.Sex == 'female']['Name'].map(getName)
names_counts = girlsNames.value_counts()
print(names_counts.head(1).index.values[0], file=answers)

answers.close()