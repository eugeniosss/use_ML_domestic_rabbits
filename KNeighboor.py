import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.ticker as ticker
import mglearn

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
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)


#plt.figure(figsize=(6,6))
#plt.scatter(X_read[y=="Domestic"]["SNP1"], X_read[y=="Domestic"]["SNP2"], color="red", label="Domestic", alpha=0.6)
#plt.scatter(X_read[y=="Wild"]["SNP1"], X_read[y=="Wild"]["SNP2"], color="blue", label="Wild", alpha=0.6)
#plt.xlabel("SNP1")
#plt.ylabel("SNP2")
#plt.legend()
#plt.show()



accs = []
ks = range(1, 21)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric='hamming')
    knn.fit(X_train, y_train_enc)
    accs.append(accuracy_score(y_test_enc, knn.predict(X_test)))

plt.plot(ks, accs, marker="o")
plt.xlabel("k (neighbors)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.xticks(ks) 
plt.show()





k=4


knn=KNeighborsClassifier(n_neighbors=k, weights="distance", metric='hamming')
knn.fit(X_train, y_train_enc)

y_pred=knn.predict(X_test)
y_pred_proba=knn.predict_proba(X_test)[:, 1]

print("Accuracy", accuracy_score(y_test_enc, y_pred))

print(classification_report(y_test_enc,y_pred, target_names=le.classes_))


cm = confusion_matrix(y_test_enc, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks([0, 1], le.classes_)
plt.yticks([0, 1], le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test_enc, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (k-NN)")
plt.legend()
plt.show()
