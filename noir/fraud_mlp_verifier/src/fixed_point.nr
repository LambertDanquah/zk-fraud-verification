pub global SCALE: Field = 100000;


// Convert SCALE² back to SCALE.
//
// Inputs:
//     x_q * w_q
//
// gives:
//
//     real * SCALE²
//
// This returns:
//
//     real * SCALE

pub fn rescale(x: Field) -> Field {

    x / SCALE

}