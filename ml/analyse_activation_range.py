import torch
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler


# =============================
# Settings
# =============================

SCALE = 100000


# =============================
# Load dataset
# =============================

df = pd.read_csv(
    "data/creditcard.csv"
)

X = df.drop(
    ["Time", "Class"],
    axis=1
)


# Match training preprocessing

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Quantise inputs

X_q = torch.tensor(
    X_scaled * SCALE,
    dtype=torch.float32
)



# =============================
# Load fused weights
# =============================

def load(name):

    with open(
        f"weights/{name}.json"
    ) as f:

        return torch.tensor(
            json.load(f),
            dtype=torch.float32
        )


W1 = load("W1")
b1 = load("b1")

W2 = load("W2")
b2 = load("b2")

W3 = load("W3")
b3 = load("b3")



# =============================
# Forward pass
# =============================

with torch.no_grad():

    # Layer 1

    z1 = (
        X_q @ W1.T
    ) + b1

    z1 = z1 / SCALE


    h1 = torch.relu(z1)


    # Layer 2

    z2 = (
        h1 @ W2.T
    ) + b2



    h2 = torch.relu(z2)


    # Output

    y = (
        h2 @ W3.T
    ) + b3



# =============================
# Print ranges
# =============================

print("\nLayer 1 before ReLU")
print("-------------------")
print("Min:", z1.min().item())
print("Max:", z1.max().item())


print("\nLayer 1 after ReLU")
print("------------------")
print("Min:", h1.min().item())
print("Max:", h1.max().item())


print("\nLayer 2 before ReLU")
print("-------------------")
print("Min:", z2.min().item())
print("Max:", z2.max().item())


print("\nLayer 2 after ReLU")
print("------------------")
print("Min:", h2.min().item())
print("Max:", h2.max().item())


print("\nOutput")
print("------")
print("Min:", y.min().item())
print("Max:", y.max().item())