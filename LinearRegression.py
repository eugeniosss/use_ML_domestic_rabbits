
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

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


def parse_train_test (X_read, y, teste_size=0.2, randome_state=0):
    X_train, X_test, y_train, y_test= train_test_split(X_read, y, test_size=teste_size, random_state=randome_state)
    return X_train, X_test, y_train, y_test

def build_LogReg(X_to_train, y_to_train):
    clf = LogisticRegression()
    clf.fit(X_to_train, y_to_train)
    return clf

def predic(model, X_to_test):
    y_pred=model.predict(X_to_test)
    return y_pred
  
def get_accuracy_of_model(y_pred, y_test):
    print("Accuracy: ", accuracy_score(y_pred, y_test))


if __name__ == "__main__":
    X,y=read_and_parse("rabbits.csv")
    X_read=only_bi(X)
    X_train, X_test, y_train, y_test= parse_train_test(X_read, y, randome_state=0)
    
    LogReg=build_LogReg(X_train, y_train)
    y_pred=predic(LogReg, X_test)
    y_pred_proba = LogReg.predict_proba(X_test)[:,1]  # probability of class "1" (Wild)
    get_accuracy_of_model(y_pred, y_test)
    print(classification_report(y_test, y_pred, target_names=["Domestic", "Wild"]))
    
    
    coeffs = pd.Series(LogReg.coef_[0], index=X_read.columns)
    # Top 10 positive and top 10 negative
    top_pos = coeffs.sort_values(ascending=False).head(10)
    top_neg = coeffs.sort_values().head(10)
    top_coeffs = pd.concat([top_pos, top_neg])



    plt.figure(figsize=(10,6))
    plt.barh(top_coeffs.index, top_coeffs.values, color=['red' if x>0 else 'blue' for x in top_coeffs.values])
    plt.xlabel("Coefficient value")
    plt.title("Top SNPs contributing to Domestic vs Wild classification")
    plt.gca().invert_yaxis()  # largest on top
    plt.show()

    plt.figure(figsize=(10,6))
    plt.hist(y_pred_proba[y_test=="Domestic"], alpha=0.5, label="Domestic", color='red')
    plt.hist(y_pred_proba[y_test=="Wild"], alpha=0.5, label="Wild", color='blue')
    plt.xlabel("Predicted probability")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.show()


    cm=confusion_matrix(y_test, y_pred)

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks([0,1], LogReg.classes_)
    plt.yticks([0,1], LogReg.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='black')

    plt.title("Confusion Matrix")
    plt.show()

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    fpr, tpr, thresholds = roc_curve(y_test_enc, y_pred_proba)

    fpr, tpr, thresholds = roc_curve(y_test_enc, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
    plt.plot(linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()