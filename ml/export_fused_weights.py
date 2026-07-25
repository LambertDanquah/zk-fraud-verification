import torch
import json
import os


os.makedirs(
    "weights",
    exist_ok=True
)


files = {
    "W1": "fused_bn_weights/W1.pt",
    "b1": "fused_bn_weights/b1.pt",

    "W2": "fused_bn_weights/W2.pt",
    "b2": "fused_bn_weights/b2.pt",

    "W3": "fused_bn_weights/W3.pt",
    "b3": "fused_bn_weights/b3.pt",
}



for name, path in files.items():

    tensor = torch.load(
        path,
        weights_only=True
    )


    data = tensor.tolist()


    with open(
        f"weights/{name}.json",
        "w"
    ) as f:

        json.dump(
            data,
            f
        )


    print(
        "Saved:",
        f"weights/{name}.json",
        tensor.shape
    )



print("\nFused weights exported successfully")