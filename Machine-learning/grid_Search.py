from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

iris=datasets.load_iris()
import pandas as pd

df = pd.DataFrame(iris.data,columns=iris.feature_names)
#print(df)

df['flower']=iris.target
df['flower']=df['flower'].apply(lambda x: iris.target_names[x])
print(df)

#Use train_test_split and manually tune parameters by trial and error
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)
model = svm.SVC(kernel='rbf',C=30,gamma='auto')
y_predicted = model.fit(X_train,y_train)
print(model.score(X_test, y_test))

#Use K Fold Cross validation

svm1 = cross_val_score(svm.SVC(kernel='linear',C=10,gamma='auto'),iris.data, iris.target, cv=5)
svm2 = cross_val_score(svm.SVC(kernel='rbf',C=10,gamma='auto'),iris.data, iris.target, cv=5)
svm3 = cross_val_score(svm.SVC(kernel='rbf',C=20,gamma='auto'),iris.data, iris.target, cv=5)
print(svm1,svm2,svm3)

def aver(p):
    y= np.average(p)
    print(y)
    

aver(svm1)
aver(svm2)
aver(svm3)

#for loop for the above steps

kernels = ['rbf', 'linear']
C = [1,10,20]
avg_scores = {}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel=kval,C=cval,gamma='auto'),iris.data, iris.target, cv=5)
        avg_scores[kval + '_' + str(cval)] = np.average(cv_scores)

print(avg_scores)

#Use GridSearchCV

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
d = pd.DataFrame(clf.cv_results_)
print("Grid search results: \n",d)

print(d[['param_C','param_kernel','mean_test_score']])
print(clf.best_params_)
print(clf.best_score_)


#Use RandomizedSearchCV to reduce number of iterations and with random combination of parameters.
#It helps reduce the cost of computation

from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, 
    cv=5, 
    return_train_score=False, 
    n_iter=4
)
rs.fit(iris.data, iris.target)
result = pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]

print("randomized CV results\n")
print(result)

    
