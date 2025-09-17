import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GridSearchCV, learning_curve
from sklearn.ensemble import GradientBoostingRegressor

df=pd.read_csv("synthetic_rabbits_quantitative.csv")

X=df.drop(labels=["Age","Latitude","Longitude"],axis=1)

y=df[["Age","Latitude","Longitude"]]

X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0, test_size=0.2)

### REGRESSION ###

##model = make_pipeline(
    ##StandardScaler(),
    ##MultiOutputRegressor(LinearRegression())
##)

##model.fit(X_train, y_train)

##y_pred=model.predict(X_test)

##y_test_np = y_test.values

##for i, col in enumerate(y.columns):
    ##mse = mean_squared_error(y_test[col], y_pred[:, i])
    ##r2 = r2_score(y_test[col], y_pred[:, i])
    ##print(f"{col}: MSE = {mse:.2f}, R2 = {r2:.2f}")

##for i, col in enumerate(y.columns):
    ##plt.scatter(y_test_np[:, i], y_pred[:, i])
    ##plt.xlabel(f"True {col}")
    ##plt.ylabel(f"Predicted {col}")
    ##plt.plot([y_test_np[:, i].min(), y_test_np[:, i].max()],
             ##[y_test_np[:, i].min(), y_test_np[:, i].max()], 'r--')
    ##plt.show()

### Gradient Boosting Regressor


params={"estimator__n_estimators": [10, 100, 500, 1000],
        "estimator__learning_rate": [0.1, 0.5, 0.7],
        "estimator__max_depth": [1, 3, 5],
        "estimator__random_state": [0]
        }

estimator = MultiOutputRegressor(GradientBoostingRegressor())

model=GridSearchCV(estimator, params, cv=5)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

y_test_np = y_test.values
y_train_np = y_train.values

for i, col in enumerate(y.columns):
    mse=mean_squared_error(y_test_np[:,i], y_pred[:,i])
    r2=r2_score(y_test_np[:,i], y_pred[:,i])
    print(f"{col}: MSE={mse:.2f}, R2={r2:.2f}")

for i, col in enumerate(y.columns):
    plt.scatter(y_test_np[:,i],y_pred[:,i])
    plt.xlabel(f"True {col}")
    plt.ylabel(f"Predicted {col}")
    plt.plot([y_test_np[:, i].min(), y_test_np[:, i].max()],
             [y_test_np[:, i].min(), y_test_np[:, i].max()], 'r--')    
    plt.show()

for i, estimator in enumerate(model.best_estimator_.estimators_):
    importances = estimator.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    print(f"Top features for {y.columns[i]}:")
    for idx in sorted_idx[:5]:
        print(f"{X_train.columns[idx]}: {importances[idx]:.3f}")


for i, col in enumerate(y.columns):
    estimator = model.best_estimator_.estimators_[i]
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X_train, y_train_np[:, i], 
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', label="Validation score")
    plt.fill_between(train_sizes, train_mean-train_scores.std(axis=1), train_mean+train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(train_sizes, test_mean-test_scores.std(axis=1), test_mean+test_scores.std(axis=1), alpha=0.1)
    
    plt.xlabel("Training samples")
    plt.ylabel("R²")
    plt.title(f"Learning curve for {col}")
    plt.legend()
    plt.show()

dump(model, "gradient_boost_multioutput.joblib")

# Later, load it back
loaded_model = load("gradient_boost_multioutput.joblib")

# Make predictions with the loaded model
y_pred_loaded = loaded_model.predict(X_test)