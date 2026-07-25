import torch
import os


# =============================
# Load trained model
# =============================

state = torch.load(
    "fraud_mlp.pth",
    weights_only=True
)


# =============================
# BatchNorm fusion function
# =============================

def fuse_linear_bn(
    W,
    b,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5
):

    """
    Fuse:

        Linear(x)=Wx+b

    followed by:

        BN(y)=gamma*(y-mean)/sqrt(var+eps)+beta


    into:

        W_new*x+b_new
    """


    scale = gamma / torch.sqrt(
        running_var + eps
    )


    W_new = (
        W *
        scale.reshape(-1,1)
    )


    b_new = (
        beta +
        (b - running_mean) *
        scale
    )


    return W_new, b_new



# =============================
# Extract layers
# =============================


W1 = state["network.0.weight"]
b1 = state["network.0.bias"]


gamma1 = state["network.1.weight"]
beta1 = state["network.1.bias"]

mean1 = state["network.1.running_mean"]
var1 = state["network.1.running_var"]



W2 = state["network.4.weight"]
b2 = state["network.4.bias"]


gamma2 = state["network.5.weight"]
beta2 = state["network.5.bias"]

mean2 = state["network.5.running_mean"]
var2 = state["network.5.running_var"]



W3 = state["network.8.weight"]
b3 = state["network.8.bias"]



# =============================
# Fuse
# =============================

W1_fused, b1_fused = fuse_linear_bn(
    W1,
    b1,
    gamma1,
    beta1,
    mean1,
    var1
)



W2_fused, b2_fused = fuse_linear_bn(
    W2,
    b2,
    gamma2,
    beta2,
    mean2,
    var2
)



# =============================
# Save fused model
# =============================

os.makedirs(
    "fused_bn_weights",
    exist_ok=True
)


torch.save(
    W1_fused,
    "fused_bn_weights/W1.pt"
)

torch.save(
    b1_fused,
    "fused_bn_weights/b1.pt"
)


torch.save(
    W2_fused,
    "fused_bn_weights/W2.pt"
)

torch.save(
    b2_fused,
    "fused_bn_weights/b2.pt"
)


torch.save(
    W3,
    "fused_bn_weights/W3.pt"
)

torch.save(
    b3,
    "fused_bn_weights/b3.pt"
)



print("BatchNorm fusion complete")

print()

print("W1:", W1_fused.shape)
print("b1:", b1_fused.shape)

print("W2:", W2_fused.shape)
print("b2:", b2_fused.shape)

print("W3:", W3.shape)
print("b3:", b3.shape)