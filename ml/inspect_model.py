import torch


state = torch.load(
    "fraud_mlp.pth",
    weights_only=True
)


for k in state.keys():

    print(k)