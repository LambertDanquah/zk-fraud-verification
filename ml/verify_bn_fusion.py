import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler



# =============================
# Load original model
# =============================

state = torch.load(
    "fraud_mlp.pth",
    weights_only=True
)



# =============================
# Load fused weights
# =============================

W1 = torch.load(
    "fused_bn_weights/W1.pt"
)

b1 = torch.load(
    "fused_bn_weights/b1.pt"
)


W2 = torch.load(
    "fused_bn_weights/W2.pt"
)

b2 = torch.load(
    "fused_bn_weights/b2.pt"
)


W3 = torch.load(
    "fused_bn_weights/W3.pt"
)

b3 = torch.load(
    "fused_bn_weights/b3.pt"
)



# =============================
# Create fused network
# =============================

class FusedMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            29,
            32
        )

        self.fc2 = nn.Linear(
            32,
            16
        )

        self.fc3 = nn.Linear(
            16,
            1
        )


    def forward(self,x):

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.fc2(x)

        x = torch.relu(x)

        x = self.fc3(x)

        return x



model = FusedMLP()


with torch.no_grad():

    model.fc1.weight.copy_(W1)
    model.fc1.bias.copy_(b1)

    model.fc2.weight.copy_(W2)
    model.fc2.bias.copy_(b2)

    model.fc3.weight.copy_(W3)
    model.fc3.bias.copy_(b3)



# =============================
# Prepare sample
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
    X_scaled[:10],
    dtype=torch.float32
)



# =============================
# Compare outputs
# =============================

model.eval()


with torch.no_grad():

    fused_output = model(sample)



print("Fused output:")
print(fused_output[:5])