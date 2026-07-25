fn main(
    x: [Field; 29],
    w: [Field; 29],
    b: Field,
    claimed_output: Field
) {


    let mut result = b;


    for i in 0..29 {

        result += x[i] * w[i];

    }


    assert(
        result == claimed_output
    );

}