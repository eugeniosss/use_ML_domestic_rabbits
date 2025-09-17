import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("synthetic_rabbits_quantitative.csv")

X=df.drop(labels=["Age","Latitude","Longitude"],axis=1)

y=df[["Age","Latitude","Longitude"]]

X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0, test_size=0.2)

class AgeCoords (nn.Module):
    def __init__ (self, n_dim):
        super().__init__()
        self.fc1=nn.Linear(n_dim,30)
        self.fc2=nn.Linear(30,10)
        self.out=nn.Linear(10,3)
    
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x = self.out(x) 
        return x

scaler_X=StandardScaler()
scaler_y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)

y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


train_dataset=TensorDataset(X_train, y_train)
train_loader=DataLoader(train_dataset, shuffle=True, batch_size=20)


model=AgeCoords(X_train.shape[1])
criterion = nn.MSELoss()   
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs=200

for i in range(epochs):
    model.train()
    running_loss=0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out=model.forward(X_batch)
        loss=criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        running_loss+=loss
    
    print(f"Epoch {i+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

with torch.no_grad():
    model.eval()
    y_out=model(X_test)

    y_out=scaler_y.inverse_transform(y_out)
    y_test=scaler_y.inverse_transform(y_test.numpy())

    for i, col in enumerate(y.columns):
        mse=mean_squared_error(y_test[:,i],y_out[:,i])
        r2=r2_score(y_test[:,i],y_out[:,i])

        print(f"For: {col}, MSE= {mse}, R2= {r2}")

        plt.figure(figsize=(3,4))
        plt.scatter(y_test[:,i],y_out[:,i])
        plt.xlabel("Real")
        plt.ylabel("Predicted")
        plt.show()