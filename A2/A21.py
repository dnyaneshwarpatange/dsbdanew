import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('academic.csv')

print("===========IsNa=============")
print(df.isna().sum())

print("===========IsNull===========")
print(df.isnull().sum())

print("===========NotNull===========")
print(df.notnull().sum())

print("============FillNa============")
df['raisedhands']=df['raisedhands'].fillna(df['raisedhands'].mean())
print(df)


# print("==========Rename==================")
# df.rename(columns={'class':'Div'})
# print(df)

print("===========DropNa============")
df.dropna(how='all')
print(df)


print("===========Z-score===========")
rh=df['raisedhands']
#print(race)
mean=np.mean(rh)
std=np.std(rh)
print("Mean:",mean)
print("Standard deviation:",std)

threshold=3
outlier=[]
for i in rh:
	z=(i-mean)/std
	if z>threshold:
		outlier.append(i)
print("Outlier :",outlier)

print("==========IQR==========")
#print(race)
rh=df['raisedhands']
#Nrace=sorted(race)
#print(Nrace)

q1,q3=np.percentile(rh,[25,75])
print("Q1,Q3:",q1,q3)

iqr=q3-q1
print("IQR:",iqr)

lower_fence=q1-(1.5*iqr)
upper_fence=q3+(1.5*iqr)
print("Lower fence,upper_fence:",lower_fence,upper_fence)

outlier=[]
for x in rh:
	if((x>upper_fence)or(x<lower_fence)):
		outlier.append(x)
print('Outliner in the dataset is',outlier)

fig=plt.figure(figsize=(10,7))
plt.boxplot(df['raisedhands'])
plt.show()	
print(df['raisedhands'])
ua=np.where(df['raisedhands']>=upper_fence)[0]
la=np.where(df['raisedhands']<=lower_fence)[0]
df.drop(index=ua,inplace=True)
df.drop(index=la,inplace=True)
print("***********After removing outliner**********")
new=df['raisedhands']
print(new)
fig=plt.figure(figsize=(10,7))
plt.boxplot(new)
plt.show()

print("**********Data Transformation**********")
df['Log_attendance']=np.log(df['Discussion'])
print('**Display dataset after data transformation**')
print(df)
