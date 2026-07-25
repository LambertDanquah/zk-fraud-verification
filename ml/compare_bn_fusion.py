import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler



# =============================
# Original model
# =============================

class OriginalMLP(nn.Module):

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



# =============================
# Fused model
# =============================

class FusedMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(29,32)

        self.fc2 = nn.Linear(32,16)

        self.fc3 = nn.Linear(16,1)


    def forward(self,x):

        x = torch.relu(
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )

        x = self.fc3(x)

        return x



# =============================
# Load original weights
# =============================

original = OriginalMLP()

original.load_state_dict(
    torch.load(
        "fraud_mlp.pth",
        weights_only=True
    )
)


original.eval()



# =============================
# Load fused weights
# =============================

fused = FusedMLP()


fused.fc1.weight.data = torch.load(
    "fused_bn_weights/W1.pt"
)

fused.fc1.bias.data = torch.load(
    "fused_bn_weights/b1.pt"
)


fused.fc2.weight.data = torch.load(
    "fused_bn_weights/W2.pt"
)

fused.fc2.bias.data = torch.load(
    "fused_bn_weights/b2.pt"
)


fused.fc3.weight.data = torch.load(
    "fused_bn_weights/W3.pt"
)

fused.fc3.bias.data = torch.load(
    "fused_bn_weights/b3.pt"
)


fused.eval()



# =============================
# Dataset sample
# =============================

df = pd.read_csv(
    "data/creditcard.csv"
)


X = df.drop(
    ["Time","Class"],
    axis=1
)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


sample = torch.tensor(
    X_scaled[:100],
    dtype=torch.float32
)



# =============================
# Compare
# =============================

with torch.no_grad():

    original_output = original(sample)

    fused_output = fused(sample)



difference = torch.abs(
    original_output -
    fused_output
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
        "SUCCESS: BatchNorm fusion preserved inference"
    )

else:

    print(
        "WARNING: Fusion changed outputs"
    )