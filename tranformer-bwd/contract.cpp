#include "contract.h"

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
) {
    for (int i = 0; i < shape_A; i++) {
        int input_idx = i * shape_B + input_offset;
        int output_idx = i * shape_C + output_offset;
        for (int j = 0; j < shape_C; j++) {
            TYPE_DATA res = 0;
            int weight_idx = j * shape_B + weight_offset;
            for (int k = 0; k < shape_B; k++) {
                res += input[input_idx + k] * weight[weight_idx + k];
            }
            output[output_idx + j] = res;
        }
    }
}