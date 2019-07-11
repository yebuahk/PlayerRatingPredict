import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df.head()

df.columns

df = df.astype(int)


df = df.fillna(df.mean())
df.isnull().any()
df.isnull().sum().sum()

X = df[['Potential', 'Acceleration', 'Aggression', 'Agility',
       'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve',
       'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving',
       'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes',
       'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing',
       'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions',
       'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed',
       'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys']]

X.head()

y = df['Overall']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)

lm.intercept_
lm.coef_
pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])

predict = lm.predict(X_test)
plt.scatter(predict, y_test)
sns.scatterplot(x=predict, y=y_test, data=df)

plt.hist(y_test - predict)
diff = y_test - predict
sns.distplot(diff)


import sklearn.metrics as mc
import  numpy as np

mc.mean_absolute_error(y_test, predict)
mc.mean_squared_error(predict, y_test)
np.sqrt(mc.mean_squared_error(predict, y_test))

y_test.to_csv('y_test_data1.csv')
p = pd.DataFrame(predict)

p.to_csv('y_predict1.csv')

plt.hist(y_test - p)