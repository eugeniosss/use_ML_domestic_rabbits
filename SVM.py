import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from scipy.stats import loguniform
import seaborn as sns
import matplotlib.pyplot as plt
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

X_train, X_test, y_train, y_test = train_test_split(X_read, y , test_size=0.2, random_state=0)
le=LabelEncoder()

y_train_enc=le.fit_transform(y_train)
y_test_enc=le.transform(y_test)

#StandardScaler()
#SVC=SVC(kernel="linear", C=1.0, probability=True, random_state=0)

#SVC.fit(X_train, y_train_enc)
#y_pred=SVC.predict(X_test)

#importance = np.abs(SVC.coef_[0])
#top_snps_idx = np.argsort(importance)[::-1][:10]  # top 10 SNPs



#fpr, tpr, thresholds = roc_curve(y_test_enc, y_pred)
#roc_auc = auc(fpr, tpr)

#plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
#plt.plot(linestyle='--', color='gray')
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.title("ROC Curve")
#plt.legend()
#plt.show()

#print("Accuracy: ", accuracy_score(y_test_enc, SVC.predict(X_test)))
#print(classification_report(y_test_enc, SVC.predict(X_test), target_names=le.classes_))

#cm=confusion_matrix(y_test_enc,SVC.predict(X_test))

#plt.figure(figsize=(4,6))
#sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#plt.xlabel("Predicted")
#plt.ylabel("Actual")
#plt.title("SVM Confusion Matrix")
#plt.xticks([0.5, 1.5], le.classes_)
#plt.yticks([0.5, 1.5], le.classes_)
#plt.show()

#kernels={"linear", "rbf", "poly"}

#results=[]

#for k in kernels:
    #svc=SVC(kernel=k, 
            #C=1.0, 
            #probability=True, 
            #random_state=0)

    #svc.fit(X_train, y_train_enc)

    #results.append({"Activation": k, "Test accuracy": accuracy_score(y_test_enc, svc.predict(X_test))})

#df=pd.DataFrame(results)

#print(df)

#Cs = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3]
#scores = []

#for C in Cs:
    #clf = make_pipeline(
        #StandardScaler(),
        #SVC(kernel="linear", C=C, class_weight=None, random_state=0)
    #)
    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    #score = cross_val_score(clf, X_train, y_train_enc, cv=cv, scoring="accuracy").mean()
    #scores.append((C, score))

#for C, s in scores:
    #print(f"C={C:g}  CV acc={s:.3f}")



pipe = make_pipeline(StandardScaler(), SVC())

# 1. GridSearchCV (exhaustive)
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train_enc)

print("Best parameters (GridSearchCV):", grid.best_params_)
print("Best CV score:", grid.best_score_)
print("Test set score:", grid.score(X_test, y_test_enc))

# 2. RandomizedSearchCV (faster for large grids)

param_dist = {
    "svc__C": loguniform(1e-3, 1e3),   # continuous distribution
    "svc__gamma": loguniform(1e-4, 1e1),
    "svc__kernel": ["linear", "rbf"]
}
random_search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=cv, scoring="accuracy", n_jobs=-1, random_state=0)
random_search.fit(X_train, y_train_enc)

print("Best parameters (RandomizedSearchCV):", random_search.best_params_)
print("Best CV score:", random_search.best_score_)
print("Test set score:", random_search.score(X_test, y_test_enc))