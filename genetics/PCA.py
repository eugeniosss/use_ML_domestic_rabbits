import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

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

scaler = StandardScaler()
X_scaled=scaler.fit_transform(X_read)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratios:", pca.explained_variance_ratio_)

plt.figure(figsize=(4,6))
plt.scatter(X_pca[y_enc==0,0], X_pca[y_enc==0, 1], label="Wild", alpha=0.7)
plt.scatter(X_pca[y_enc==1, 0], X_pca[y_enc==1, 1], label="Domestic", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Rabbit SNPs")
plt.legend()
plt.show()


loadings = pd.DataFrame(pca.components_.T, index=X_read.columns, columns=["PC1", "PC2"])

top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(5)
print("Top 5 SNPs for PC1:\n", top_pc1)

# Top 5 SNPs contributing to PC2
top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(5)
print("Top 5 SNPs for PC2:\n", top_pc2)