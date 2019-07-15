import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df.head()

df.columns

# df = df.astype(int)


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

y = df['Overall']

X.shape
y.shape


# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)

lm.score(X_test, y_test)

lm.intercept_
lm.coef_

''' applying Ridge to penalize coefficient or number of features --- L2'''
from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.01, normalize=True)
ridgeReg.fit(X_train, y_train)

pred = ridgeReg.predict(X_test)

ridgeReg.score(X_test, y_test)


from sklearn.linear_model import Lasso
lassoReg = Lasso(alpha=0.01, normalize=True)
lassoReg.fit(X_train, y_train)

pred_l = lassoReg.predict(X_test)

lassoReg.score(X_test, y_test)


'''y = c + mx where m is your co-efficient and c is your intercept '''
pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])

predict = lm.predict(X_test)
plt.scatter(predict, y_test)
sns.scatterplot(x=predict, y=y_test, data=df)


plt.plot(X_test, predict, color='blue')

# plt.hist(y_test - predict)
# diff = y_test - predict
# sns.distplot(diff)

import sklearn.metrics as mc

mc.mean_squared_error(y_test, pred)
mc.mean_squared_error(y_test, pred_l)

mc.mean_absolute_error(y_test, predict)
mc.mean_squared_error(predict, y_test)
np.sqrt(mc.mean_squared_error(predict, y_test))

y_test.to_csv('y_test_data111.csv')
p = pd.DataFrame(pred)

p.to_csv('y_predict111.csv')

plt.hist(y_test - pred)
plt.scatter(y_test, pred)
sns.scatterplot(y_test, pred)
#
# yy = y_test.to_list()
# pp = pd.Series(pred).to_list()
#
# plt.scatter(yy, pp)