#ifndef ORDER_CONTROL
#define ORDER_CONTROL

#include "defines.h"
#include "contract.h"

void order_control_tt_grad(
    TYPE_WEIGHT* tt_cores,
    int* tt_ranks,
    int* tt_shapes,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    TYPE_WEIGHT* grad_cores
);


#endif 
