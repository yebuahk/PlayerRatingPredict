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
#
# age_position = df['Preferred Positions'].groupby(df['Age'])
# ap = pd.DataFrame(age_position)
# ap

# df_string['Nationality'] = df_string['Nationality'].astype('category')  # convert to categorical for memory
df['Nationality'] = df_string['Nationality'].astype('category')     # convert to categorical for memory
# df.Nationality = df.Nationality.astype('category') #alternaitive as above convert to category



df['Age'] = pd.to_numeric(df['Age'], errors = 'coerce')   #  account for any nan in Age column

"""Function to Convert categorical data to integers. Note: you can use encoder or pandas dummy as well"""

def  recode_position(position):
    if position == 'ST':
        return 1
    elif position == 'CB':
        return 0
    else:
        return np.nan

# apply function to recode position
df.columns
df['recode'] = df['PreferredPositions'].apply(recode_position)  # apply function

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

""" fill null values with mean of specific column values"""
# eng = df.loc[df['Nationality'] == 'England']
# nm = eng['Age'].mean()
#
# df.loc[df['Nationality'] == 'England'].fillna(nm)
#
# df.dropna()
# df.to_csv('c_fifa.csv')

from sklearn.model_selection import train_test_split
train, validation = train_test_split(df, test_size=0.30, random_state=5)

""" 70/50 split of data for test and validation"""
train.shape
validation.shape

""" will use below to train smaller data set"""
# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# loo.split(df)
#
# df.head()
#
# X = df.iloc[:, 1:]   # all rows with columns starting from 2
# X.head()
#
# y = df['Overall']
# y.head()
#
# for train_index, test_index in loo.split(X):
#     print('train : ', train_index, 'test : ', test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

from sklearn.model_selection import KFold
X = df.iloc[:, 1:]   # all rows with columns starting from 2
y = df['Overall']

kf = KFold(n_splits=5, random_state=None, shuffle=False)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    print('train : ', train_index, 'test : ', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


from sklearn.model_selection import StratifiedKFold
X = df.iloc[:, 1:]   # all rows with columns starting from 2
y = df['Overall']

skf = StratifiedKFold(n_splits=5, random_state=None)
skf.get_n_splits(X)

for train_index, test_index in skf.split(X, y):
    print('train : ', train_index, 'test : ', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]