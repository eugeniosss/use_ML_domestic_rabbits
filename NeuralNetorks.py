import pandas as pd
import seaborn as sns
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

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

X_train, X_test, y_train, y_test = train_test_split(X_read, y, random_state=0, test_size=0.2)

le=LabelEncoder()
y_train_enc=le.fit_transform(y_train)
y_test_enc=le.transform(y_test)


mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),  # 1 hidden layer, 50 neurons
                    activation="relu", 
                    solver="adam", 
                    max_iter=1000,
                    random_state=0)

mlp.fit(X_train, y_train_enc)

y_pred=mlp.predict(X_test)

print("Accuracy", accuracy_score(y_test_enc, y_pred))
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))



cm=confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - MLPClassifier")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(mlp.loss_curve_, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MLP Training Loss Curve")
plt.show()

