mod neuron;
mod layer;



fn main(

    x: [Field;29],

    W1: [[Field;29];32],
    b1: [Field;32],

    W2: [[Field;32];16],
    b2: [Field;16],

    W3: [[Field;16];1],
    b3: [Field;1],

    claimed: [Field;1]

) {


    let h1 = layer::dense_29_32(
        x,
        W1,
        b1
    );


    let h2 = layer::dense_32_16(
        h1,
        W2,
        b2
    );


    let output = layer::dense_16_1(
        h2,
        W3,
        b3
    );


    assert(
        output[0] == claimed[0]
    );

}