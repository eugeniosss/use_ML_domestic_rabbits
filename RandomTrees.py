import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

X_train, X_test, y_train, y_test=train_test_split(X_read, y, random_state=0, test_size=0.2)

le=LabelEncoder()

y_train_encoded=le.fit_transform(y_train)
y_test_encoded=le.transform(y_test)

rf=RandomForestClassifier(n_estimators=100,max_depth=None, random_state=0, oob_score=True)
rf.fit(X_train, y_train_encoded)

y_pred=rf.predict(X_test)
y_pred_proba=rf.predict_proba(X_test)[:,1]

print("Accuracy: ", accuracy_score(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

cm=confusion_matrix(y_test_encoded, y_pred)

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

fpr, tpr, _ = roc_curve(y_test_encoded, y_pred)
roc_auc=auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

importances = pd.Series(rf.feature_importances_, index=X_read.columns)
top_importances = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
plt.barh(top_importances.index, top_importances.values, color="green")
plt.xlabel("Feature importance")
plt.title("Top SNPs contributing to classification")
plt.gca().invert_yaxis()
plt.show()

tree = rf.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(tree,
          feature_names=X_read.columns,
          class_names=le.classes_,
          filled=True,
          rounded=True,
          fontsize=10)
plt.show()