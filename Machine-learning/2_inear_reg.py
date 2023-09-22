import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n

df=pd.read_csv("C:/Users/Priyanka/Desktop/pythonds/hiring.csv")
print(df)

df.experience = df.experience.fillna('zero')
print(df)

df.experience = df.experience.apply(w2n.word_to_num)
print(df)

import math
median_test_score = math.floor(df['test_score(out of 10)'].mean())
print(median_test_score)

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])

t = reg.predict([[2,9,6]])
f = reg.predict([[12,10,10]])
print(t,f)
