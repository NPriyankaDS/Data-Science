import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_csv("C:/Users/Priyanka/Desktop/pythonds/titanic.csv")
print(df.sample(5))
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head())

inputs = df.drop('Survived',axis='columns')
target = df.Survived

inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
print(inputs.sample(100))

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.sample(100))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)
print(len(X_train))
print(len(X_test))

#training the model
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)

#prediction using the model
y_predicted=model.predict(X_test)
print(y_predicted)

print(model.score(X_test,y_test))

