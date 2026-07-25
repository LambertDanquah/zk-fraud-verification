# ============================
# Fixed-point ReLU lookup
#
# SCALE = 100000
#
# Input:
#   real activation * SCALE
#
# Example:
#   1.25 -> 125000
#
# ============================


LOOKUP_MIN = -5000000
LOOKUP_MAX = 5000000



def relu_lookup(x):

    x = int(x)


    if x < LOOKUP_MIN or x > LOOKUP_MAX:

        raise ValueError(
            f"Activation {x} outside lookup range "
            f"[{LOOKUP_MIN}, {LOOKUP_MAX}]"
        )


    # ReLU:
    #
    # negative values -> 0
    # positive values -> unchanged

    if x < 0:
        return 0


    return x