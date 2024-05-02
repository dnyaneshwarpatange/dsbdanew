import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset
#print(sns.get_dataset_names())
df=sns.load_dataset('titanic')
#print(df)
tips=load_dataset("tips")

# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
#---------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display and fill the Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
df['age'].fillna(df['age'].median())
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
#One variable
fig, axes = plt.subplots(1,2)
sns.boxplot(data = df, y ='age', ax=axes[0])
sns.boxplot(data = df, y ='fare', ax=axes[1])
plt.show()
# Two variables
fig, axes = plt.subplots(1,3, sharey=True)
sns.boxplot(data = df, x='sex', y ='age', hue = 'sex', ax=axes[0])
sns.boxplot(data = df, x='pclass', y ='age', hue = 'pclass', ax=axes[1])
sns.boxplot(data = df, x='survived', y ='age', hue = 'survived', ax=axes[2])
plt.show()
# Two variables
fig, axes = plt.subplots(1,3, sharey=True)
sns.boxplot(data = df, x='sex', y ='fare', hue = 'sex', ax=axes[0], log_scale = True)
sns.boxplot(data = df, x='pclass', y ='fare', hue = 'pclass', ax=axes[1], log_scale = True)
sns.boxplot(data = df, x='survived', y ='fare', hue = 'survived', ax=axes[2], log_scale = True)
plt.show()
#three variables
fig, axes = plt.subplots(1,2, sharey=True)
sns.boxplot(data = df, x='sex', y ='age', hue = 'survived', ax=axes[0])
sns.boxplot(data = df, x='pclass', y ='age', hue = 'survived', ax=axes[1])
plt.show()
fig, axes = plt.subplots(1,2, sharey=True)
sns.boxplot(data = df, x='sex', y ='fare', hue = 'survived', ax=axes[0], log_scale = True)
sns.boxplot(data = df, x='pclass', y ='fare', hue = 'survived', ax=axes[1], log_scale = True)
plt.show()
