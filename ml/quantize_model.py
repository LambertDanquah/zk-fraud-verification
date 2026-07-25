import torch
import json
import os


# =============================
# Load fused weights
# =============================

weights = torch.load(
    "fused_weights.pth",
    weights_only=True
)


# =============================
# Quantization settings
# =============================

SCALE = 100000


print("Using scale:", SCALE)



# =============================
# Quantization function
# =============================

def quantize(tensor):

    return torch.round(
        tensor * SCALE
    ).to(torch.int64)



# =============================
# Create output directory
# =============================

os.makedirs(
    "weights",
    exist_ok=True
)



# =============================
# Quantize parameters
# =============================

quantized = {}


for name, value in weights.items():

    q_value = quantize(value)

    quantized[name] = q_value


    print(
        name,
        value.shape,
        "->",
        q_value.shape
    )



# =============================
# Save JSON files
# =============================

for name, tensor in quantized.items():

    path = f"weights/{name}.json"


    with open(path,"w") as f:

        json.dump(
            tensor.tolist(),
            f
        )


    print(
        "Saved:",
        path
    )



# Save scale

with open(
    "weights/scale.json",
    "w"
) as f:

    json.dump(
        SCALE,
        f
    )


print("\nQuantization complete")