import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

def read_and_parse(file):
    df = pd.read_csv(file)
    X=df.drop("Label", axis=1)
    y=df["Label"]
    return X,y

def only_bi (X_in):
    biallelic_snps=[]

    for column in X_in.columns:
        uniques=X_in[column].unique()
        if len(uniques)==2:
            biallelic_snps.append(column)

    X_biallelic=X_in[biallelic_snps]
    X_encoded=pd.DataFrame()

    for col in X_biallelic.columns:
        ref = X_biallelic[col].value_counts().idxmax()
        X_encoded[col] = (X_biallelic[col] != ref).astype(int)
    return X_encoded

X,y=read_and_parse("rabbits_multisnp.csv")
X_read=only_bi(X)

le=LabelEncoder()
y_enc=le.fit_transform(y)

kmeans = KMeans(n_clusters=2, random_state=0)
clusters=kmeans.fit_predict(X_read)

#print("Cluster assignments:", clusters[:20])  # first 20

comparison = pd.crosstab(y_enc, clusters, rownames=["True"], colnames=["Cluster"])
print(comparison)

#from sklearn.metrics import silhouette_score

#inertias = []
#silhouettes = []
#K = range(2, 8)
#for k in K:
    #km = KMeans(n_clusters=k, random_state=0).fit(X)
    #inertias.append(km.inertia_)
    #silhouettes.append(silhouette_score(X, km.labels_))

#plt.plot(K, inertias, "-o")
#plt.xlabel("k")
#plt.ylabel("Inertia")
#plt.title("Elbow method")
#plt.show()

#plt.plot(K, silhouettes, "-o")
#plt.xlabel("k")
#plt.ylabel("Silhouette score")
#plt.title("Silhouette analysis")
#plt.show()


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(X_read)  # X_read is your encoded SNP dataframe

# Soft assignments
probs = gmm.predict_proba(X_read)  # shape: (n_samples, n_clusters)

# Hard assignments (for comparison)
labels = gmm.predict(X_read)

#print("Soft membership probabilities (first 5 samples):\n", probs[:5])
#print("Hard cluster assignments (first 20 samples):\n", labels[:20])
comparison = pd.crosstab(y_enc, labels, rownames=["True"], colnames=["Cluster"])
print(comparison)

plt.figure(figsize=(8,4))
plt.bar(range(len(probs)), probs[:,0], color="skyblue", label="Cluster 0")
plt.bar(range(len(probs)), probs[:,1], bottom=probs[:,0], color="salmon", label="Cluster 1")
plt.xlabel("Sample index")
plt.ylabel("Cluster membership probability")
plt.title("Soft cluster membership (GMM)")
plt.legend()
plt.show()