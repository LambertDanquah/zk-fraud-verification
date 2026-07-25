import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler



# =========================
# Original model
# =========================

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



# =========================
# Load original model
# =========================

original = FraudMLP()

original.load_state_dict(
    torch.load(
        "fraud_mlp.pth",
        weights_only=True
    )
)

original.eval()



# =========================
# Load fused weights
# =========================

weights = torch.load(
    "fused_weights.pth",
    weights_only=True
)



# =========================
# Fused forward
# =========================

def fused_forward(x):

    W1 = weights["W1"]
    b1 = weights["b1"]

    W2 = weights["W2"]
    b2 = weights["b2"]

    W3 = weights["W3"]
    b3 = weights["b3"]


    h1 = torch.matmul(
        x,
        W1.T
    ) + b1


    h1 = torch.relu(h1)


    h2 = torch.matmul(
        h1,
        W2.T
    ) + b2


    h2 = torch.relu(h2)


    out = torch.matmul(
        h2,
        W3.T
    ) + b3


    return out



# =========================
# Test random samples
# =========================

df = pd.read_csv(
    "data/creditcard.csv"
)


X = df.drop(
    ["Class","Time"],
    axis=1
)


scaler = StandardScaler()

X = scaler.fit_transform(X)


samples = torch.tensor(
    X[:100],
    dtype=torch.float32
)



with torch.no_grad():

    original_output = original(samples)

    fused_output = fused_forward(samples)



difference = torch.abs(
    original_output - fused_output
)



print(
    "Maximum difference:",
    difference.max().item()
)


print(
    "Average difference:",
    difference.mean().item()
)


if difference.max() < 1e-5:

    print(
        "SUCCESS: Models are equivalent"
    )

else:

    print(
        "WARNING: Difference detected"
    )