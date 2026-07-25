import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =====================================
# Settings
# =====================================

SCALE = 100000


# BN254 scalar field modulus used by Noir/Barretenberg

FIELD_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


def to_field(value):

    """
    Convert signed integer into Noir Field representation.
    """

    if value < 0:
        return FIELD_MODULUS + value

    return value



# =====================================
# Load dataset
# =====================================

df = pd.read_csv(
    "data/creditcard.csv"
)



# Remove timestamp and label

X = df.drop(
    ["Time", "Class"],
    axis=1
)



# =====================================
# Normalize features
# Must match training preprocessing
# =====================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)



# Take first transaction

x = X_scaled[0]



# =====================================
# Load first neuron weights
# =====================================

with open(
    "weights/W1.json"
) as f:

    W1 = json.load(f)



with open(
    "weights/b1.json"
) as f:

    b1 = json.load(f)



# First neuron of first hidden layer

w = W1[0]

bias = b1[0]



# =====================================
# Quantize input
# =====================================

x_q = [
    round(v * SCALE)
    for v in x
]



# W1 and b1 are already quantized
# from quantize_model.py

w_q = w

b_q = round(
    bias
)



# =====================================
# Compute neuron output
# =====================================

z = b_q


for i in range(29):

    z += (
        x_q[i] *
        w_q[i]
    )



z_q = round(z)



# =====================================
# Generate Noir Prover.toml
# =====================================

output_path = (
    "../noir/fraud_mlp_verifier/Prover.toml"
)



with open(
    output_path,
    "w"
) as f:


    # Input vector

    f.write(
        "x = [\n"
    )


    for value in x_q:
        f.write(
            f'"{to_field(value)}",\n'
            )


    f.write(
        "]\n\n"
    )



    # Weight vector

    f.write(
        "w = [\n"
    )


    for value in w_q:
        f.write(
            f'"{to_field(value)}",\n'
            )


    f.write(
        "]\n\n"
    )



    # Bias

    f.write(
        f'b = "{to_field(b_q)}"\n\n'
    )



    # Claimed neuron output

    f.write(
        f'claimed_output = "{to_field(z_q)}"\n'
    )



print("Generated Prover.toml")

print("---------------------")

print("First neuron output:")
print(z_q)

print()

print("Saved to:")
print(output_path)