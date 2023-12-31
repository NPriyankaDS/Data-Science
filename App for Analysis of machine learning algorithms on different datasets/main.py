import streamlit as st 
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title('Analysis of machine learning algorithms on different datasets')
st.write("""
# Explore different classifiers
Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select classifier",("KNN","SVM","Random forest"))

@st.cache_data
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    
    X = data.data
    y = data.target
    return X,y

X,y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))

# prepare a summary report of the accuracy of various models on the datasets

def add_parameter_ui(clf_name):
    #global K, C, max_depth, n_estimators
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)
st.write("params:\n" ,params)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif clf_name == "SVM":
        clf = SVC(C = params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                    random_state=1234)
    return clf

clf = get_classifier(classifier_name,params)

# Classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")
if st.button("classification report"):
    st.text("report:\n " + report)


summary = {'dataset': dataset_name,'classifer': classifier_name,'hyperparameter':params,
            'accuracy': acc}
df_summary = pd.DataFrame(summary)
    
table = st.dataframe(df_summary)
#table.add_rows(df_summary)



#Plot the graphs

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha = 0.8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show()

st.pyplot(fig)