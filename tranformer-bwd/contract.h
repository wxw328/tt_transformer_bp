#ifndef CONTRACT
#define CONTRACT

#include "defines.h"

void contract(
    TYPE_DATA* input,
    TYPE_DATA* output,
    TYPE_WEIGHT* weight,
    int weight_offset,
    int input_offset,
    int output_offset,
    int shape_A,
    int shape_B,
    int shape_C
);

#endif

