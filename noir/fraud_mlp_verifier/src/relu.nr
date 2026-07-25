// =============================
// Mask-based ReLU
//
// mask = 0  -> output = 0
// mask = 1  -> output = input
//
// Assumes mask ∈ {0,1}.
// =============================

pub fn relu(
    x: Field,
    mask: Field
) -> Field {

    // Boolean constraint
    assert(mask * (mask - 1) == 0);

    x * mask
}