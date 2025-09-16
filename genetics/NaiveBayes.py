import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

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

X_train, X_test, y_train, y_test=train_test_split(X_read, y, test_size=0.2, random_state=0)

le=LabelEncoder()
y_train_enc=le.fit_transform(y_train)
y_test_enc=le.fit_transform(y_test)

##models = {
    ##"BernoulliNB": BernoulliNB(),
    ##"MultinomialNB": MultinomialNB(),
    ##"GaussianNB": GaussianNB()
##}

##for name, model in models.items():
    ##model.fit(X_train, y_train)
    ##y_pred=model.predict(X_test)
    ##test_acc = accuracy_score(y_test, y_pred)
    ##print(f"{name} Test Accuracy: {test_acc:.3f}")

    ##cm=confusion_matrix(y_test, y_pred)

    ##plt.figure(figsize=(4,6))
    ##sns.heatmap(cm, annot=True, xticklabels=["Wild","Domestic"], yticklabels=["Wild","Domestic"])
    ##plt.xlabel("Predicted")
    ##plt.ylabel("True")
    ##plt.title("Naive Bayes Confusion Matrix")
    ##plt.show()



nb=GaussianNB()
nb.fit(X_train, y_train_enc)

y_pred=nb.predict(X_test)


means = nb.theta_      # shape: (n_classes, n_features)
variances = nb.var_    # shape: (n_classes, n_features)

# Importance estimate: absolute difference between class means, normalized by variance
importance = np.abs(means[0] - means[1]) / np.sqrt(variances.mean(axis=0))

# Rank SNPs by importance
top_features = np.argsort(importance)[::-1][:10]  # top 10
print("Top SNP indices:", top_features)
print("Importance scores:", importance[top_features])

# Top 5 most important SNPs
top_features = np.argsort(importance)[::-1][:5]

# Bottom 5 least important SNPs
least_features = np.argsort(importance)[:5]

all_features = np.concatenate([top_features, least_features])
titles = [f"Top {i+1} SNP {snp}" for i, snp in enumerate(top_features)] + \
         [f"Least {i+1} SNP {snp}" for i, snp in enumerate(least_features)]

plt.figure(figsize=(20, 8))

for i, snp in enumerate(all_features):
    plt.subplot(2, 5, i+1)
    sns.kdeplot(X_train.loc[y_train_enc==0, X_train.columns[snp]], label="Wild", fill=True, alpha=0.5)
    sns.kdeplot(X_train.loc[y_train_enc==1, X_train.columns[snp]], label="Domestic", fill=True, alpha=0.5)
    plt.title(titles[i])
    plt.xlabel("SNP Value")
    plt.ylabel("Density")
    if i == 0:  # only show legend once
        plt.legend()

plt.tight_layout()
plt.show()

scores = cross_val_score(nb, X_train, y_train_enc, cv=5)
print("CV mean accuracy:", scores.mean())

plt.figure(figsize=(4,6))
fpr, tpr, _ =roc_curve(y_test_enc, y_pred)
roc_auc=auc(fpr, tpr)
plt.plot(fpr, tpr,label=f"AUC = {roc_auc:.2f}", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()