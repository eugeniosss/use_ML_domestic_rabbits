import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

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

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

# 1. GridSearchCV
param_grid = {
    "logisticregression__penalty": ["l1", "l2"],
    "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    "logisticregression__solver": ["liblinear", "saga"]
}

grid=GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train_enc)

y_pred=grid.predict(X_test)
print("Accuracy: ", accuracy_score(y_pred, y_test_enc))

cm=confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(4,6))
sn.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.xticks([0.5, 1.5], le.classes_)
plt.yticks([0.5, 1.5], le.classes_)
plt.show()