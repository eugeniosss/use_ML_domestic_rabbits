import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
y_train_enc=le.fit_transform(y_train)
y_test_enc=le.transform(y_test)

# Convert numpy → torch
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_enc, dtype=torch.float32).view(-1,1)  # make it (n_samples,1)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.float32).view(-1,1)

# Wrap in dataset + dataloader (for batching)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class SNPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SNPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)  # binary output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # output between 0-1
        return x

# Initialize model
model = SNPClassifier(input_dim=X_train.shape[1])

# Loss function & optimizer
criterion = nn.BCELoss()  # Binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50  # number of times we pass through the entire dataset

for epoch in range(n_epochs):
    model.train()  # put model in training mode
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()          # reset gradients from previous batch
        outputs = model(X_batch)       # forward pass
        loss = criterion(outputs, y_batch)  # compute loss
        loss.backward()                # backpropagation
        optimizer.step()               # update weights

        running_loss += loss.item()    # accumulate batch loss

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(train_loader):.4f}")

model.eval()  # set model to evaluation mode
with torch.no_grad():  # no need to track gradients
    y_pred = model(X_test_tensor).round()   # 0 or 1
    accuracy = (y_pred == y_test_tensor).float().mean().item()

print(f"Test Accuracy: {accuracy:.4f}")

# Save
torch.save(model.state_dict(), "snp_model.pth")

# Later, reload
model = SNPClassifier(input_dim=X_train.shape[1])
model.load_state_dict(torch.load("snp_model.pth"))
model.eval()  # important: set to evaluation mode