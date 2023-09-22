import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df)

df1=df[df.target==0]
df2=df[df.target==1]
df3=df[df.target==2]

#print(df1,df2,df3)

       
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='b',marker='*')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='g',marker='.')
plt.scatter(4.8,3.0 ,color='yellow',marker='*')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="b",marker='+')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'],color="g",marker='.')
plt.scatter(1.5,0.3,color='yellow',marker='*')
plt.show()


X = df.drop(['target'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Create KNN (K Neighrest Neighbour Classifier)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
print(knn.predict([[4.8,3.0,1.5,0.3]]))


#Plot Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#Print classification report for precesion, recall and f1-score for each classes

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
