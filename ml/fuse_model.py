import torch
import torch.nn as nn


# -----------------------------
# Same architecture as training
# -----------------------------

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



# -----------------------------
# Load trained model
# -----------------------------

model = FraudMLP()


model.load_state_dict(
    torch.load(
        "fraud_mlp.pth",
        weights_only=True
    )
)


model.eval()



# -----------------------------
# BatchNorm fusion function
# -----------------------------

def fuse_linear_bn(linear, bn):

    W = linear.weight.data
    b = linear.bias.data


    gamma = bn.weight.data
    beta = bn.bias.data

    mean = bn.running_mean
    var = bn.running_var

    eps = bn.eps


    scale = gamma / torch.sqrt(
        var + eps
    )


    W_fused = W * scale.unsqueeze(1)


    b_fused = beta + (
        b - mean
    ) * scale


    return W_fused, b_fused



# -----------------------------
# Extract layers
# -----------------------------

fc1 = model.network[0]
bn1 = model.network[1]

fc2 = model.network[4]
bn2 = model.network[5]

fc3 = model.network[8]



# Fuse BatchNorm

W1,b1 = fuse_linear_bn(
    fc1,
    bn1
)


W2,b2 = fuse_linear_bn(
    fc2,
    bn2
)


W3 = fc3.weight.data
b3 = fc3.bias.data



# -----------------------------
# Save fused weights
# -----------------------------

torch.save(
    {
        "W1": W1,
        "b1": b1,

        "W2": W2,
        "b2": b2,

        "W3": W3,
        "b3": b3
    },

    "fused_weights.pth"
)


print("Fusion complete")
print("Saved fused_weights.pth")


print("\nShapes:")
print("W1:", W1.shape)
print("b1:", b1.shape)

print("W2:", W2.shape)
print("b2:", b2.shape)

print("W3:", W3.shape)
print("b3:", b3.shape)