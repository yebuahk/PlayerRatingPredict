"""author Kevin Yebuah 7.6.2019
This is to predict player rating
based on features
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('datafifa.csv')

"""data pre-processing and insights"""

df.columns
df.head(10) # take a look at first 10 rows
df.describe()
df.info()
df.shape    # 17981 rows and 36 columns


df['Overall'].value_counts(dropna=False)    # count number of values in column e.g. 13 players have 83 rating
df['Finishing'].value_counts(dropna=True)   # set dropna to True to include n/a in count

df.dtypes   # data types check
df.get_dtype_counts()   # data types count

df_string = df.select_dtypes(include=['object'])    # separate any stings
df_string.shape
df_numerical = df.select_dtypes(exclude=['object'])     # separate any numerical
df_numerical.shape  # same rows and columns as df.shape meaning all columns contain integers

df_numerical.info()

df['Nationality'] = df_string['Nationality'].astype('category')     # convert to categorical for memory

df['Age'] = pd.to_numeric(df['Age'], errors = 'coerce')   #  account for any nan in Age column


# apply function to recode position
df.columns
# df['recode'] = df['PreferredPositions'].apply(recode_position)  # apply function

""" using lambda function to add """
df['Aging'] = df['Age'].apply(lambda x: x+1)

""" Replace CDM CM with CDM = Center defensive midfield"""
df['newPosn'] = df['PreferredPositions'].apply(lambda x: x.replace('CDM CM', 'CDM'))

df.drop_duplicates()    # to drop duplicates

""" checking for null values"""
df.isnull().values.any()
df.isnull().any()
df.isnull().sum()
df.isnull().sum().sum()

df['Acceleration2'] = df['Acceleration'].fillna(df['Acceleration'].mean())  # replace null values with mean

df.columns

df.drop(['Name', 'Nationality', 'PreferredPositions'], axis=1)
X = df.iloc[:, 1:]   # all rows with columns starting from 2
X.shape
y = df['Overall']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

lm.intercept_
lm.coef_
pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])
