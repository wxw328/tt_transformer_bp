#ifndef FC_BWD
#define FC_BWD

#include "defines.h"
#include "contract.h"

void compute_tt_grad(
    float* tt_cores,
    int* tt_ranks,
    int* tt_shapes,
    int input_dims,
    int output_dims,
    float* input,
    float* output,
    float* grad_output,
    float* grad_input,
    float* grad_cores
);

#endif 

