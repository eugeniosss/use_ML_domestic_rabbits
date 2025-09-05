import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier

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

params={"n_estimators": [10, 100, 500, 1000],
        "learning_rate": [0.1, 0.5, 0.7],
        "max_depth": [1, 3, 5],
        "random_state": [0]
        }


pipe = GradientBoostingClassifier()

grid=GridSearchCV(pipe,param_grid=params,cv=5)
grid.fit(X_train, y_train_enc)

print("Best parameters (GridSearchCV):", grid.best_params_)
print("Best CV score:", grid.best_score_)
print("Test set score:", grid.score(X_test, y_test_enc))

results_df = pd.DataFrame(grid.cv_results_)

# Show relevant columns
results_df = results_df[[
    "params", "mean_test_score", "std_test_score", "rank_test_score","param_n_estimators", "param_learning_rate","param_max_depth"
]]

# Sort by rank or mean_test_score
results_df = results_df.sort_values("rank_test_score")
print(results_df.head(10))  # top 10 parameter combinations

for depth in sorted(results_df["param_max_depth"].unique()):
    subset = results_df[results_df["param_max_depth"] == depth]
    heatmap_data = subset.pivot(
        index="param_n_estimators",
        columns="param_learning_rate",
        values="mean_test_score"
    )
    plt.figure(figsize=(6,4))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"CV Accuracy (max_depth={depth})")
    plt.ylabel("param_n_estimators")
    plt.xlabel("param_learning_rate")
    plt.show()

gbt = grid.best_estimator_

feature_importance=pd.Series(gbt.feature_importances_, index=gbt.feature_names_in_)
feature_importance=feature_importance.sort_values(ascending=False)

top_snps=feature_importance.head(10)
top_snps.plot(kind="bar",figsize=(4,6))
plt.title("Most important SNPs")
plt.ylabel("Importance")
plt.show()