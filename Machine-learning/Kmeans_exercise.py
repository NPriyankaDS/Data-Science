from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

iris=load_iris()
print(dir(iris))

df = pd.DataFrame(iris.data,columns=iris.feature_names)
#print(df.head())


df=df.drop(['sepal length (cm)','sepal width (cm)'],axis='columns')
print(df.head())

df['target']=iris.target
print(df)

ax = plt.axes(projection ="3d")
ax.scatter3D(df['petal length (cm)'],df['petal width (cm)'],df['target'])
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.clabel('target')
plt.show()


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['petal length (cm)','petal width (cm)','target']])
#print(y_predicted)

df['cluster']=y_predicted
print(df.sample(15))
print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
ax = plt.axes(projection ="3d")
ax.scatter3D(df1['petal length (cm)'],df1['petal width (cm)'],df1['target'],color='g')
ax.scatter3D(df2['petal length (cm)'],df2['petal width (cm)'],df2['target'],color='b')
ax.scatter3D(df3['petal length (cm)'],df3['petal width (cm)'],df3['target'],color='r')

ax.scatter3D(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2],color='purple',marker='*',label='centroid')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.clabel('target')
plt.legend()
plt.show()

#Preprocessing using min max scaler
scaler = MinMaxScaler()
scaler.fit(df[['petal length (cm)']])
df['petal length (cm)']=scaler.transform(df[['petal length (cm)']])
scaler.fit(df[['petal width (cm)']])
df['petal width (cm)']=scaler.transform(df[['petal width (cm)']])
scaler.fit(df[['target']])
df['target']=scaler.transform(df[['target']])
print(df)                           
   
                                            
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['petal length (cm)','petal width (cm)','target']])
df['cluster']=y_predicted
print(df.sample(15))
print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
ax = plt.axes(projection ="3d")
ax.scatter3D(df1['petal length (cm)'],df1['petal width (cm)'],df1['target'],color='g')
ax.scatter3D(df2['petal length (cm)'],df2['petal width (cm)'],df2['target'],color='b')
ax.scatter3D(df3['petal length (cm)'],df3['petal width (cm)'],df3['target'],color='r')

ax.scatter3D(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2],color='purple',marker='*',label='centroid')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.clabel('target')
plt.legend()
plt.show()

#elbow plot to determine optimal value of k 

sse = []
k_rng = range(1,20)
for k in k_rng:
    km = KMeans(n_clusters=k,n_init='auto')
    km.fit(df[['petal length (cm)','petal width (cm)','target']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()
