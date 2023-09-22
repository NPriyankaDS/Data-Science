import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("C:/Users/Priyanka/Desktop/pythonds/diabetes.csv")
print(df)
print(df.isnull().sum())
print(df.describe())
print(df.Outcome.value_counts())


#train test split
X = df.drop("Outcome",axis="columns")
y = df.Outcome
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled, y, stratify=y, random_state=10)
#y_traindf=pd.DataFrame(y_train)
#y_testdf=pd.DataFrame(y_test)
print(len(X_train))
print(len(y_train))

#print(y_traindf.value_counts())
#print(y_testdf.value_counts())

#Train using stand alone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
print(scores.mean())

#Train using Bagging

from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
bag_model.fit(X_train, y_train)
print(bag_model.oob_score_)
print(bag_model.score(X_test, y_test))

#calculate the score using K FOLD CROSS validation

scores_bag = cross_val_score(bag_model, X, y, cv=5)
print(scores_bag.mean())

#We can see some improvement in test score with bagging classifier as compared to a standalone classifier
#Train using Random Forest

from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=5)
print(scores.mean())
