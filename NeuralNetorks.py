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

### TEST ACTIVATIONS ###

#activations = ["relu", "tanh", "logistic"]
#results = []

#for act in activations:
    #mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),  # 1 hidden layer, 50 neurons
                    #activation=act, 
                    #solver="adam", 
                    #max_iter=1000,
                    #random_state=0)
    #mlp.fit(X_train, y_train_enc)
    #train_accuracy=accuracy_score(y_train_enc, mlp.predict(X_train))
    #test_accuracy=accuracy_score(y_test_enc, mlp.predict(X_test))

    #results.append({"Activation": act, "Train accuracy": train_accuracy, "Test accuracy": test_accuracy})

#df=pd.DataFrame(results)

#plt.figure(figsize=(6,4))
#sns.barplot(x="Activation", y="Test accuracy", data=df)
#plt.title("Comparison of Activation Functions (Test Accuracy)")
#plt.show()

### TEXT HIDDEN LAYERS AND NEURON

#structures = [(100), (100,50), (100,50,20), (100,100,100), (200,100,50, 20), (100,100,100,100), (100,100,100,100,100)]
#results = []

#for str in structures:
    #mlp = MLPClassifier(hidden_layer_sizes=str,  # 1 hidden layer, 50 neurons
                    #activation="relu", 
                    #solver="adam", 
                    #max_iter=1000,
                    #random_state=0)
    #mlp.fit(X_train, y_train_enc)
    #train_accuracy=accuracy_score(y_train_enc, mlp.predict(X_train))
    #test_accuracy=accuracy_score(y_test_enc, mlp.predict(X_test))

    #results.append({"Structure": str, "Train accuracy": train_accuracy, "Test accuracy": test_accuracy})

#df=pd.DataFrame(results)

#print(df)

#plt.figure(figsize=(6,4))
#sns.barplot(x="Structure", y="Train accuracy", data=df)
#plt.title("Comparison of Different structures (Test Accuracy)")
#plt.show()

### TEXT SOLVERS

#solvers = ["adam", "lbfgs","sgd"]
#results = []

#for solver in solvers:
    #mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100),  # 1 hidden layer, 50 neurons
                    #activation="relu", 
                    #solver=solver, 
                    #max_iter=1000,
                    #random_state=0)
    #mlp.fit(X_train, y_train_enc)
    #train_accuracy=accuracy_score(y_train_enc, mlp.predict(X_train))
    #test_accuracy=accuracy_score(y_test_enc, mlp.predict(X_test))

    #results.append({"Solver": solver, "Train accuracy": train_accuracy, "Test accuracy": test_accuracy})

#df=pd.DataFrame(results)

#print(df)

#plt.figure(figsize=(6,4))
#sns.barplot(x="Solver", y="Train accuracy", data=df)
#plt.title("Comparison of Different structures (Test Accuracy)")
#plt.show()

### TEXT ALPHAS

#plt.figure(figsize=(8,6))

#alphas = [0.05, 0.01, 0.001, 0.0001]
#results = []

#for alpha in alphas:
    #mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100),  # 1 hidden layer, 50 neurons
                    #activation="relu", 
                    #olver="adam", 
                    #max_iter=1000,
                    #random_state=0,
                    #alpha=alpha)
    #mlp.fit(X_train, y_train_enc)
    #train_accuracy=accuracy_score(y_train_enc, mlp.predict(X_train))
    #test_accuracy=accuracy_score(y_test_enc, mlp.predict(X_test))

    #results.append({"Alpha": alpha, "Train accuracy": train_accuracy, "Test accuracy": test_accuracy})

    #sns.lineplot(x=range(len(mlp.loss_curve_)), 
                 #y=mlp.loss_curve_, 
                 #label=f"alpha={alpha}")

#plt.xlabel("Iteration")
#plt.ylabel("Training Loss")
#plt.title("MLP Training Loss Curves for Different Alphas")
#plt.legend()
#plt.show()

#df=pd.DataFrame(results)

#print(df)

#plt.figure(figsize=(6,4))
#sns.barplot(x="Alpha", y="Train accuracy", data=df)
#plt.title("Comparison of Different alphas (Test Accuracy)")
#plt.show()

### TEST EVERYTHING

activations = ["relu", "tanh", "logistic"]
structures = [(100), (100,50), (100,50,20), (100,100,100), (200,100,50, 20), (100,100,100,100), (100,100,100,100,100)]
solvers = ["adam", "lbfgs","sgd"]
alphas = [0.05, 0.01, 0.001, 0.0001]
results = []

for act in activations:
    for str in structures:
        for solver in solvers:
            for alpha in alphas:
                mlp = MLPClassifier(hidden_layer_sizes=str,
                    activation=act, 
                    solver=solver, 
                    max_iter=1000,
                    random_state=0,
                    alpha=alpha)
                
                mlp.fit(X_train, y_train_enc)
                
                train_accuracy=accuracy_score(y_train_enc, mlp.predict(X_train))
                test_accuracy=accuracy_score(y_test_enc, mlp.predict(X_test))
                
                results.append({"Alpha": alpha, 
                                "Structure": str,
                                "Solver": solver,
                                "Activation": act,
                                "Train accuracy": train_accuracy, 
                                "Test accuracy": test_accuracy})

df=pd.DataFrame(results)

df.sort_values("Train accuracy")



#mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),  # 1 hidden layer, 50 neurons
                    #activation="relu", 
                    #solver="adam", 
                    #max_iter=1000,
                    #random_state=0)

#mlp.fit(X_train, y_train_enc)

#y_pred=mlp.predict(X_test)

#print("Accuracy", accuracy_score(y_test_enc, y_pred))
#print(classification_report(y_test_enc, y_pred, target_names=le.classes_))



#cm=confusion_matrix(y_test_enc, y_pred)
#plt.figure(figsize=(6,4))
#sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
#plt.xlabel("Predicted")
#plt.ylabel("True")
#plt.title("Confusion Matrix - MLPClassifier")
#plt.show()

#plt.figure(figsize=(6,4))
#plt.plot(mlp.loss_curve_, marker="o")
#lt.xlabel("Iteration")
#plt.ylabel("Loss")
#plt.title("MLP Training Loss Curve")
#plt.show()

