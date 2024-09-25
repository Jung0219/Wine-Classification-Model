# Softmax and cross entropy
# dropout and batch normalization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math


class ClassificationModel(nn.Module):
    def __init__(self, n_features, n_hidden_layers1, n_hidden_layers2, n_labels):
        # two hidden layers
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden_layers1)
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(n_hidden_layers1)
        self.fc2 = nn.Linear(n_hidden_layers1, n_hidden_layers2)
        self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(n_hidden_layers2)
        self.fc3 = nn.Linear(n_hidden_layers2, n_labels)

    def forward(self, data):
        data = self.fc1(data)
        data = self.bn1(data)
        data = F.relu(data)
        data = self.dropout1(data)
        data = self.fc2(data)
        data = self.bn2(data)
        data = F.relu(data)
        data = self.dropout2(data)
        data = self.fc3(data)
        return data
        # no softmax!


class WineDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = pd.DataFrame(x).values
        self.y = pd.DataFrame(y).values
        self.n_features = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        features, labels = self.x[index], self.y[index]
        if self.transform:
            features, labels = self.transform(features), self.transform(labels)
        return features, torch.tensor(labels, dtype=torch.long).squeeze()

    def __len__(self):
        return self.n_features


class ToTensor():
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float32)


model = ClassificationModel(13, 32, 16, 3)
n_epochs = 30
scalar = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(
    scalar.fit_transform(load_wine().data), load_wine().target, train_size=0.8)

train_data = WineDataset(x_train, y_train, transform=ToTensor())

train_dataloader = DataLoader(train_data, batch_size=5, shuffle=True)


criterion = nn.CrossEntropyLoss()  # wants long
optimizer = optim.SGD(model.parameters(), lr=0.002)

model.train()
for epoch in range(n_epochs):
    avg_loss = 0
    for features, labels in train_dataloader:
        # forward
        predictions = model(features)
        loss = criterion(predictions, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss
    avg_loss /= n_epochs
    if epoch % 2 == 0:
        print(f"loss:{avg_loss:.4f}")

test_data = WineDataset(x_test, y_test, ToTensor())
test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)


model.eval()
total = 0
correct = 0


with torch.no_grad():
    for features, labels in test_dataloader:
        predictions = model(features)
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
