from turtle import mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("synthetic_rabbits_quantitative.csv")

X=df.drop(labels=["Age","Latitude","Longitude"],axis=1)

y_Age=df[["Age"]]

X_scaler=StandardScaler()

X_torch=torch.FloatTensor(X_scaler.fit_transform(X))

y_Age_scaler=StandardScaler()

y_Age_torch=torch.FloatTensor(y_Age_scaler.fit_transform(y_Age))

X_train_full, X_test, y_train_full, y_test = train_test_split(X_torch, y_Age_torch, test_size=0.2, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

training_dataset=TensorDataset(X_train, y_train)

training_loader=DataLoader(training_dataset, batch_size=24, shuffle=True)

class PredictAge (nn.Module):
    def __init__ (self, n_dims):
        super().__init__()
        self.fc1=nn.Linear(n_dims, 32)
        self.bn1 = nn.BatchNorm1d(32)     
        self.fc2=nn.Linear(32,16)
        self.bn2 = nn.BatchNorm1d(16)     
        self.out=nn.Linear(16,1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x=self.out(x)
        return x

model_Age=PredictAge(X_train.shape[1])

optimizer = optim.Adam(model_Age.parameters(), lr=0.0001)
criterion=nn.MSELoss()

epochs=1000

best_val_loss=999999
patiente=30
counter=0

for i in range(epochs):
    model_Age.train()
    running_loss=0

    for X_batch, y_batch in training_loader:
        optimizer.zero_grad()
        
        y_pred=model_Age.forward(X_batch)
        
        loss=criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    
    if (i+1) % 10==0:
        print(f"Epoch {i+1}, loss = {running_loss/len(training_loader):.4f}")
    
    with torch.no_grad():
        model_Age.eval()
        y_pred=model_Age(X_val)
        loss=criterion(y_val, y_pred).item()
    
    if loss < best_val_loss:
        counter=0
        best_val_loss = loss
    
    else:
        counter+=1
        if counter>=patiente:
            print(f"Early stopping at eppoch {i+1}")
            break


with torch.no_grad():
    model_Age.eval()
    y_pred=y_Age_scaler.inverse_transform(model_Age(X_test))
    MSE=mean_squared_error(y_Age_scaler.inverse_transform(y_test),y_pred)
    r2=r2_score(y_Age_scaler.inverse_transform(y_test), y_pred)
    print(f"MSE={MSE:.4f} and R2={r2:.4f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_Age_scaler.inverse_transform(y_test), y_pred)
    plt.xlabel("real values")
    plt.ylabel("predicted values")
    plt.plot([y_pred.min(),y_pred.max()],
             [y_pred.min(), y_pred.max()], 'r--'
    ) 
    plt.show()

y_coords=df[["Latitude","Longitude"]]

y_coords_scaler=StandardScaler()

y_coords_torch=torch.FloatTensor(y_coords_scaler.fit_transform(y_coords))

X_train_full, X_test, y_train_full, y_test = train_test_split(X_torch, y_coords_torch, test_size=0.2, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

TenDat=TensorDataset(X_train, y_train)
DatLoa=DataLoader(TenDat, batch_size=24, shuffle=True)

class PredictCoords(nn.Module):
    def __init__ (self, n_dims):
        super().__init__()
        self.fc1 = nn.Linear(n_dims, 32)
        self.bc1=nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bc2=nn.BatchNorm1d(16)
        self.out = nn.Linear(16, 2)
    def forward(self,x):
        x=torch.relu(self.bc1(self.fc1(x)))
        x=torch.relu(self.bc2(self.fc2(x)))
        x=self.out(x)
        return x

modelCoords=PredictCoords(X.shape[1])

optimizer=optim.Adam(modelCoords.parameters(),lr=0.0001)
crite=nn.MSELoss()

epochs=50000

best_val_loss = 99999
patience = 30        # how many epochs to wait for improvement
counter = 0

for epoch in range(epochs):
    modelCoords.train()
    running_loss = 0

    for X_batch, y_batch in DatLoa:
        optimizer.zero_grad()
        y_pred = modelCoords(X_batch)
        loss = crite(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Compute validation loss
    modelCoords.eval()
    with torch.no_grad():
        val_pred = modelCoords(X_val)
        val_loss = crite(val_pred, y_val).item()
    
    print(f"Epoch {epoch+1}, Train Loss={running_loss/len(DatLoa):.4f}, Val Loss={val_loss:.4f}")
    
    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

with torch.no_grad():
    modelCoords.eval()
    y_pred = y_coords_scaler.inverse_transform(modelCoords(X_test))
    y_test = y_coords_scaler.inverse_transform(y_test)

    n_targets = y_coords.shape[1]  # how many columns you are predicting

    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 6))  # 1 row, N columns

    if n_targets == 1:
        axes = [axes]  # make it iterable if there's only one target

    for i, col in enumerate(y_coords.columns):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"For {col}, MSE={mse:.4f} and R2={r2:.4f}")

        ax = axes[i]
        ax.scatter(y_test[:, i], y_pred[:, i])
        ax.set_xlabel("Real")
        ax.set_ylabel("Predicted")
        ax.set_title(col)

        # Add diagonal "perfect prediction" line
        min_val = min(y_test[:, i].min().item(), y_pred[:, i].min().item())
        max_val = max(y_test[:, i].max().item(), y_pred[:, i].max().item())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.tight_layout()
    plt.show()