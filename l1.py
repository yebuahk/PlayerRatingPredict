import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('USA_Housing.csv')
df.head()

df.info()
df.columns


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

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
sns.scatterplot(predict, y_test, hue=predict)

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

t = pd.DataFrame(y_test)

j = pd.concat([p, t], axis=1)
j.head()