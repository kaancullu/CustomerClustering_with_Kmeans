import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv("customers.csv")

X = data[["Hacim","Maas"]].values


scores = []
for i in range(1,11):
    kmns = KMeans(n_clusters=i, init="k-means++", random_state=123)
    kmns.fit(X)
    scores.append(kmns.inertia_) #WCSS values
    
s = plt.plot(scores, color="purple")
plt.title("Scores For Number Of Clusters")


kmns = KMeans(n_clusters=3, init="k-means++")
kmns.fit(X)
pred = kmns.fit_predict(X).astype(str)

print("Cluster Centers\n----------------------------------\n")
print(kmns.cluster_centers_)

letters = ["A","B","C","D"]
for i,v in enumerate(pred):
    pred[i]=letters[int(v)] 

datas = data[["Hacim","Maas"]]
datas["Predict"] = pred

datas1 = datas.sort_values("Predict")

#VISUALIZATION
facet = sns.lmplot(data=datas1, x='Hacim', y='Maas', hue='Predict', fit_reg=False, legend=True, legend_out=True)
