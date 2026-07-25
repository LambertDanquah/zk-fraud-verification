import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================
# Settings
# =========================

SCALE = 100000

FIELD_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


def to_field(value):

    value = int(round(value))

    if value < 0:
        value = FIELD_MODULUS + value

    if value > 2147483647:
        return f'"{value}"'

    return str(value)


# =========================
# Load dataset
# =========================

df = pd.read_csv("data/creditcard.csv")

X = df.drop(
    ["Time", "Class"],
    axis=1
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

x = X_scaled[0]

# Input quantisation
x_q = [
    int(round(v * SCALE))
    for v in x
]


# =========================
# Load weights
# =========================

def load(name):

    with open(f"weights/{name}.json") as f:
        return json.load(f)


W1 = load("W1")
b1 = load("b1")

W2 = load("W2")
b2 = load("b2")

W3 = load("W3")
b3 = load("b3")


# =========================
# Quantise parameters
# =========================

def quantise_matrix(M):

    return [
        [
            int(round(v * SCALE))
            for v in row
        ]
        for row in M
    ]


def quantise_vector(v):

    return [
        int(round(x * SCALE))
        for x in v
    ]


W1 = quantise_matrix(W1)
b1 = quantise_vector(b1)

W2 = quantise_matrix(W2)
b2 = quantise_vector(b2)

W3 = quantise_matrix(W3)
b3 = quantise_vector(b3)


# =========================
# Forward Pass
#
# IMPORTANT:
#
# ReLU implemented in Python only.
#
# Also exports activation masks.
#
# =========================


# =========================
# Layer 1
# 29 -> 32
# =========================

h1 = []
h1_z = []
h1_mask = []

for n in range(32):

    z = b1[n]

    for i in range(29):

        z += (
            x_q[i] *
            W1[n][i]
        )

    # Store pre-ReLU activation
    h1_z.append(z)

    # Activation mask
    h1_mask.append(
        1 if z > 0 else 0
    )

    # ReLU output
    h1.append(
        max(0, z)
    )


print("\nLayer 1")
print("First 5:", h1[:5])
print("Max:", max(h1))
print("Min:", min(h1))
print("Active:", sum(h1_mask))


# =========================
# Layer 2
# 32 -> 16
# =========================

h2 = []
h2_z = []
h2_mask = []

for n in range(16):

    z = b2[n]

    for i in range(32):

        z += (
            h1[i] *
            W2[n][i]
        )

    # Store pre-ReLU activation
    h2_z.append(z)

    # Activation mask
    h2_mask.append(
        1 if z > 0 else 0
    )

    # ReLU output
    h2.append(
        max(0, z)
    )


print("\nLayer 2")
print("First 5:", h2[:5])
print("Max:", max(h2))
print("Min:", min(h2))
print("Active:", sum(h2_mask))


# =========================
# Output layer
# =========================

y = b3[0]

for i in range(16):

    y += (
        h2[i] *
        W3[0][i]
    )


print("\nOutput")
print(y)

claimed = [y]


# =========================
# Generate Prover.toml
# =========================

path = "../noir/fraud_mlp_verifier/Prover.toml"

with open(path, "w") as f:

    # Input
    f.write("x = [\n")

    for v in x_q:
        f.write(f"{to_field(v)},\n")

    f.write("]\n\n")

    # Weights
    for name, data in [
        ("W1", W1),
        ("W2", W2),
        ("W3", W3)
    ]:

        f.write(f"{name} = [\n")

        for row in data:

            f.write("[\n")

            for v in row:
                f.write(f"{to_field(v)},\n")

            f.write("],\n")

        f.write("]\n\n")

    # Biases
    for name, data in [
        ("b1", b1),
        ("b2", b2),
        ("b3", b3)
    ]:

        f.write(f"{name} = [\n")

        for v in data:
            f.write(f"{to_field(v)},\n")

        f.write("]\n\n")

    # Layer 1 activation mask
    f.write("h1_mask = [\n")

    for v in h1_mask:
        f.write(f"{to_field(v)},\n")

    f.write("]\n\n")

    # Layer 2 activation mask
    f.write("h2_mask = [\n")

    for v in h2_mask:
        f.write(f"{to_field(v)},\n")

    f.write("]\n\n")

    # Claimed output
    f.write("claimed = [\n")
    f.write(f"{to_field(y)},\n")
    f.write("]\n")


print("\nGenerated full MLP Prover.toml")
print("Architecture:")
print("29 → 32 → 16 → 1")
print("Claimed output:")
print(y)