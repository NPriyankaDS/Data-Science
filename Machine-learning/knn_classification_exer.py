import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

digits=load_digits()
df=pd.DataFrame(digits.data,columns=digits.feature_names)
df['target']=digits.target
#print(df.head())
#print(df)

df1=df[df.target==0]
df2=df[df.target==1]
print(df1)
print(df2)

X = df.drop(['target'], axis='columns')
X = np.array(X)
print(X)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Create KNN (K Neighrest Neighbour Classifier)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(knn.score(X_test, y_test))

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
