import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

df=pd.read_csv("C:/Users/Priyanka/Desktop/pythonds/heart.csv")
print(df)

#check if there are any null or na values in the dataframe
print(df.isnull().sum())
print(df.isna().any())
print(df.describe())

#check for any outliers in the data

#remove outliers
for x in (df.Age,df.RestingBP,df.Cholesterol,df.MaxHR,df.Oldpeak):
    z = np.abs(stats.zscore(x))
    print(z)



    
    

        



