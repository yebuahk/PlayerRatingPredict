import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data.csv')

df.head(10)

df.isnull().any()
df.isnull().sum().sum()

df.columns

X = df[['int.rate', 'installment', 'log.annual.inc',
       'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
       'inq.last.6mths', 'delinq.2yrs', 'pub.rec', ]]
X
y = df['pub.rec']
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

classifier.coef_
classifier.intercept_


y_pred = classifier.predict(pd.DataFrame(X_test))
y_pred

y_pred1 = classifier.predict_proba(pd.DataFrame(X_test))
y_pred1 = pd.DataFrame(y_pred1)
y_pred1.columns = ['paid', 'not paid']
