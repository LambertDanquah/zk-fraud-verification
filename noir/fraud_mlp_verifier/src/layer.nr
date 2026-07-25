use crate::neuron;



// =============================
// Layer: 29 → 32
// =============================

pub fn dense_29_32(
    x: [Field;29],
    W: [[Field;29];32],
    b: [Field;32]
) -> [Field;32] {


    let mut outputs = [0;32];


    for n in 0..32 {

        outputs[n] = neuron::compute_29(
            x,
            W[n],
            b[n]
        );

    }


    outputs
}



// =============================
// Layer: 32 → 16
// =============================

pub fn dense_32_16(
    x: [Field;32],
    W: [[Field;32];16],
    b: [Field;16]
) -> [Field;16] {


    let mut outputs = [0;16];


    for n in 0..16 {

        outputs[n] = neuron::compute_32(
            x,
            W[n],
            b[n]
        );

    }


    outputs
}



// =============================
// Layer: 16 → 1
// =============================

pub fn dense_16_1(
    x: [Field;16],
    W: [[Field;16];1],
    b: [Field;1]
) -> [Field;1] {


    let mut outputs = [0;1];


    for n in 0..1 {

        outputs[n] = neuron::compute_16(
            x,
            W[n],
            b[n]
        );

    }


    outputs
}