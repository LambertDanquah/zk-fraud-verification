import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(
    "data/creditcard.csv"
)

print("Original shape:")
print(df.shape)


# =====================================================
# Feature / Label Separation
# =====================================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


# Remove timestamp

X = X.drop(
    "Time",
    axis=1
)


# =====================================================
# Feature Normalization
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining data:")
print(X_train.shape)

print("Test data:")
print(X_test.shape)

print("Fraud examples:")
print(y_train.sum())


# =====================================================
# Convert to PyTorch tensors
# =====================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)


# Convert pandas Series -> numpy -> tensor

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1,1)


y_test = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1,1)



# =====================================================
# DataLoader
# =====================================================

dataset = TensorDataset(
    X_train,
    y_train
)


loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)



# =====================================================
# MLP Architecture
#
# 29 -> 32 -> 16 -> 1
#
# Linear
# BatchNorm
# ReLU
# Dropout
#
# Linear
# BatchNorm
# ReLU
# Dropout
#
# Linear
#
# Output = logit
# =====================================================

class FraudMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(29,32),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.Dropout(0.2),


            nn.Linear(32,16),

            nn.BatchNorm1d(16),

            nn.ReLU(),

            nn.Dropout(0.2),


            nn.Linear(16,1)

        )


    def forward(self,x):

        return self.network(x)



model = FraudMLP()



# =====================================================
# Handle Class Imbalance
#
# Fraud cases are rare.
# Give higher penalty to missing fraud.
# =====================================================

fraud_count = y_train.sum()

normal_count = len(y_train) - fraud_count


pos_weight = normal_count / fraud_count


criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



# =====================================================
# Training
# =====================================================

epochs = 50


for epoch in range(epochs):

    model.train()

    running_loss = 0


    for X_batch, y_batch in loader:


        optimizer.zero_grad()


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch
        )


        loss.backward()


        optimizer.step()


        running_loss += loss.item()



    print(
        f"Epoch {epoch+1}/{epochs}, Loss={running_loss:.4f}"
    )



# =====================================================
# Evaluation
# =====================================================

model.eval()


with torch.no_grad():

    logits = model(
        X_test
    )

    probabilities = torch.sigmoid(
        logits
    )


predictions = (
    probabilities > 0.5
).float()



y_true = y_test.numpy()

y_prob = probabilities.numpy()

y_pred = predictions.numpy()



auc = roc_auc_score(
    y_true,
    y_prob
)


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred
)


recall = recall_score(
    y_true,
    y_pred
)


f1 = f1_score(
    y_true,
    y_pred
)



print("\n==========================")
print("Evaluation Results")
print("==========================")

print(
    f"AUC: {auc:.4f}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1 Score: {f1:.4f}"
)


print(
    classification_report(
        y_true,
        y_pred
    )
)



# =====================================================
# Save Model
# =====================================================

torch.save(
    model.state_dict(),
    "fraud_mlp.pth"
)


print("\nModel saved as fraud_mlp.pth")