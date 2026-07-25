pub fn compute_29(
    x: [Field; 29],
    w: [Field; 29],
    b: Field
) -> Field {

    let mut result = b;

    for i in 0..29 {

        result += x[i] * w[i];

    }

    result
}



pub fn compute_32(
    x: [Field; 32],
    w: [Field; 32],
    b: Field
) -> Field {

    let mut result = b;

    for i in 0..32 {

        result += x[i] * w[i];

    }

    result
}



pub fn compute_16(
    x: [Field; 16],
    w: [Field; 16],
    b: Field
) -> Field {

    let mut result = b;

    for i in 0..16 {

        result += x[i] * w[i];

    }

    result
}