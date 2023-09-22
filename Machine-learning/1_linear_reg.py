import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

d = pd.read_csv("C:/Users/Priyanka/Desktop/pythonds/canada_income.csv")
print(d.head(10))
plt.xlabel('year')
plt.ylabel('per capita income(US$)')
plt.scatter(d["year"],d["per capita income (US$)"],color='red',marker='+')
#plt.show()

#applying linear regression model to the data
reg = linear_model.LinearRegression()
reg.fit(d[['year']],d['per capita income (US$)'])
plt.plot(d.year,reg.predict(d[['year']]),color='blue')
plt.show()
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[2020]]))


                        
