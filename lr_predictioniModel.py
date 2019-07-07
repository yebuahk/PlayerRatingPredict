#author Kevin Yebuah 7.6.2019
# This is to predict player rating
# based on features

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('datafifa.csv')

#data preprocessing and insights

df.columns
df.head(10)  #take a look at first 10 rows
df.describe()
df.info()
df.shape   #17981 rows and 36 columns


df['Overall'].value_counts(dropna=False) #count number of values in column e.g. 13 players have 83 rating
df['Finishing'].value_counts(dropna=True) #set dropna to True to include n/a in count

df.dtypes   #datatypes check
df.get_dtype_counts() #datatypes count

df_string = df.select_dtypes(include=['object']) #seperate any stings
df_string.shape
df_numerical = df.select_dtypes(exclude=['object']) #seperate any numerical
df_numerical.shape   #same rows and columns as df.shape meaning all columns contain integers

df_numerical.info()
#
# age_position = df['Preferred Positions'].groupby(df['Age'])
# ap = pd.DataFrame(age_position)
# ap

df_string['Nationality'] = df_string['Nationality'].astype('category') #convert to categorical for memory
df['Nationality'] = df_string['Nationality'].astype('category') #convert to categorical for memory
#df.Nationality = df.Nationality.astype('category') #alternaitive as above convert to category

df['Age'] = pd.to_numeric(df['Age'], errors='coerce') #account for any nan in Age column

#Convert categorical data to integers
def recode_position(position):
    if position == 'ST':
        return 1
    elif position == 'CB':
        return 0
    else:
        return np.nan

#apply function to recode position
df.columns
df['recode'] = df['PreferredPositions'].apply(recode_position) #apply function


