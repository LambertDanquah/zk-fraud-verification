import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================
# Settings
# =========================

SCALE = 100000


# BN254 scalar field modulus (Noir Field)

FIELD_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)



# =========================
# Convert signed integer
# to Noir Field representation
# =========================

def to_field(value):

    value = int(value)

    if value < 0:
        return f'"{FIELD_MODULUS + value}"'

    return str(value)



# =========================
# Load dataset
# =========================

df = pd.read_csv(
    "data/creditcard.csv"
)


X = df.drop(
    ["Time", "Class"],
    axis=1
)



# =========================
# Match training preprocessing
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# Use first transaction

x = X_scaled[0]



# =========================
# Quantize input
# =========================

x_q = [
    int(round(v * SCALE))
    for v in x
]



# =========================
# Load first-layer weights
# =========================

with open("weights/W1.json") as f:
    W1 = json.load(f)


with open("weights/b1.json") as f:
    b1 = json.load(f)



# =========================
# Compute first hidden layer
# (signed integer arithmetic)
# =========================

outputs = []


for neuron in range(32):

    z = int(b1[neuron])

    for i in range(29):

        z += (
            int(x_q[i]) *
            int(W1[neuron][i])
        )

    outputs.append(z)



# =========================
# Generate Noir Prover.toml
# =========================

with open(
    "../noir/fraud_mlp_verifier/Prover.toml",
    "w"
) as f:


    # ---------------------
    # Inputs
    # ---------------------

    f.write("x = [\n")

    for value in x_q:

        f.write(
            f"{to_field(value)},\n"
        )

    f.write("]\n\n")



    # ---------------------
    # Weight matrix
    # ---------------------

    f.write("W = [\n")


    for row in W1:

        f.write("[\n")


        for value in row:

            f.write(
                f"{to_field(value)},\n"
            )


        f.write("],\n")


    f.write("]\n\n")



    # ---------------------
    # Bias vector
    # ---------------------

    f.write("b = [\n")


    for value in b1:

        f.write(
            f"{to_field(value)},\n"
        )


    f.write("]\n\n")



    # ---------------------
    # Claimed outputs
    # ---------------------

    f.write("claimed = [\n")


    for value in outputs:

        f.write(
            f"{to_field(value)},\n"
        )


    f.write("]\n")



print("Layer 1 Prover.toml generated")
print()

print("Input shape : 29")
print("Weight shape: 32 x 29")
print("Bias shape  : 32")
print()

print("First 5 signed outputs:")
print(outputs[:5])

print()

print("First 5 field outputs:")
print(
    [
        to_field(v)
        for v in outputs[:5]
    ]
)