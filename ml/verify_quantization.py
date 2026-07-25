import torch
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler



# =====================================
# Load fused floating-point weights
# =====================================

float_weights = torch.load(
    "fused_weights.pth",
    weights_only=True
)



# =====================================
# Load quantized weights
# =====================================

def load_json(path):

    with open(path,"r") as f:
        return torch.tensor(
            json.load(f),
            dtype=torch.float32
        )



scale = 100000



W1_q = load_json("weights/W1.json")
b1_q = load_json("weights/b1.json")

W2_q = load_json("weights/W2.json")
b2_q = load_json("weights/b2.json")

W3_q = load_json("weights/W3.json")
b3_q = load_json("weights/b3.json")



# Convert back to approximate floats

W1_q /= scale
b1_q /= scale

W2_q /= scale
b2_q /= scale

W3_q /= scale
b3_q /= scale



# =====================================
# Forward functions
# =====================================

def forward(weights, x):

    W1,b1,W2,b2,W3,b3 = weights


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



float_output = forward(
    (
        float_weights["W1"],
        float_weights["b1"],
        float_weights["W2"],
        float_weights["b2"],
        float_weights["W3"],
        float_weights["b3"],
    ),
    None
) if False else None



# =====================================
# Load sample data
# =====================================

df = pd.read_csv(
    "data/creditcard.csv"
)


X = df.drop(
    ["Time","Class"],
    axis=1
)


scaler = StandardScaler()

X = scaler.fit_transform(X)



samples = torch.tensor(
    X[:100],
    dtype=torch.float32
)



float_output = forward(
    (
        float_weights["W1"],
        float_weights["b1"],
        float_weights["W2"],
        float_weights["b2"],
        float_weights["W3"],
        float_weights["b3"],
    ),
    samples
)



quant_output = forward(
    (
        W1_q,
        b1_q,
        W2_q,
        b2_q,
        W3_q,
        b3_q,
    ),
    samples
)



difference = torch.abs(
    float_output - quant_output
)



print(
    "Maximum difference:",
    difference.max().item()
)


print(
    "Average difference:",
    difference.mean().item()
)



if difference.max() < 1e-3:

    print(
        "SUCCESS: Quantization error acceptable"
    )

else:

    print(
        "WARNING: Quantization changed outputs significantly"
    )