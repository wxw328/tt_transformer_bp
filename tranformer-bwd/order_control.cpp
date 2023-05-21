#include "order_control.h"

/*
shape = [16,8,8,8,8,16]
rank = [1,16,30,30,30,16,1]
*/

void order_control_tt_grad(
/*
tt_cores: weights represented by TT
tt_ranks: [r_0,r_1,...,r_d+1]
tt_shapes: [i_1, i_2,...i_m, j_1,j_2,...,j_n], n + m =d
input: a vector with shape i_1*i_2*...*i_m
grad_output: a vector calculated by last layer with shape j_1*j_2*...*j_n
grad_cores: a arrary stores the result for the gradient to each core
*/
    TYPE_WEIGHT* tt_cores,
    int* tt_ranks,
    int* tt_shapes,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    TYPE_WEIGHT* grad_cores
) {
    //buffer size depends on max intermediate results, function as ping-pong buffer initialized on FPGA
    float buffer_right[2][30 * 8 * 8 * 16];
    float buffer_left[2][30 * 8 * 8 * 16];
    int in = 0;
    int out = 1;

    //caculate G1G2G3Xdy
    contract(tt_cores, buffer_right[in], tt_cores, WD * 1, 0, 0, tt_shapes[0], tt_ranks[1], tt_shapes[1] * tt_ranks[2]);
    contract(buffer_right[in], buffer_right[out], tt_cores, WD * 2, 0, 0, tt_shapes[0]*tt_shapes[1], tt_ranks[2], tt_shapes[2] * tt_ranks[3]);
    contract(buffer_right[out], buffer_right[in], input, 0, 0, 0,tt_ranks[3], tt_shapes[0] * tt_shapes[1] * tt_shapes[2], 1);
    contract(buffer_right[in], buffer_right[out], grad_output, 0, 0, 0, tt_ranks[3], 1, tt_shapes[3] * tt_shapes[4] * tt_shapes[5]);

    contract(buffer_right[out], buffer_right[in], tt_cores, WD * 3, 0, 0, tt_shapes[4] * tt_shapes[5], tt_ranks[3] * tt_shapes[3], tt_ranks[4]);
    contract(buffer_right[in], grad_cores, tt_cores, WD * 4, 0, WD * 5, tt_shapes[5], tt_ranks[4] * tt_shapes[4], tt_ranks[5]);//G6
    contract(buffer_right[in], grad_cores, tt_cores, WD * 5, 0, WD * 4, tt_ranks[4] * tt_shapes[4],tt_shapes[5], tt_ranks[5]);//G5
    float temp_right[30 * 8 * 16];
    contract(tt_cores, temp_right, tt_cores, WD * 5, WD * 4, 0, tt_ranks[4] * tt_shapes[4], tt_ranks[5], tt_shapes[5]);
    contract(buffer_right[out], grad_cores, temp_right, 0, 0, WD * 3, tt_ranks[3] * tt_shapes[3], tt_shapes[4] * tt_shapes[5], tt_ranks[4]);//G4


    //caculate G4G5G6dyX
    contract(tt_cores, buffer_left[in], tt_cores, WD * 4, WD * 3, 0, tt_ranks[3] * tt_shapes[3], tt_ranks[4], tt_shapes[4] * tt_ranks[5]);
    contract(buffer_left[in], buffer_left[out], tt_cores, WD * 5, 0, 0, tt_ranks[3] * tt_shapes[3]*tt_shapes[4], tt_ranks[5], tt_shapes[5] * tt_ranks[6]);
    contract(buffer_left[out], buffer_left[in], grad_output, 0, 0, 0, tt_ranks[3], tt_shapes[3] * tt_shapes[4] * tt_shapes[5], 1);
    contract(buffer_left[in], buffer_left[out], input, 0, 0, 0, tt_ranks[3], 1, tt_shapes[0] * tt_shapes[1] * tt_shapes[2]);

    contract(buffer_left[out], buffer_left[in], tt_cores, 0, 0, 0, tt_shapes[1] * tt_shapes[2] * tt_ranks[3], tt_shapes[0], tt_ranks[1]);
    contract(buffer_left[in], grad_cores, tt_cores, WD * 1, 0, WD * 2, tt_shapes[2] * tt_ranks[3], tt_shapes[1] * tt_ranks[1], tt_ranks[2]);//G3
    contract(buffer_left[in], grad_cores, tt_cores, WD * 2, 0, WD * 1, tt_shapes[1] * tt_ranks[1], tt_shapes[2] * tt_ranks[3], tt_ranks[2]); //G2
    float temp_left[16 * 8 * 30];
    contract(buffer_left[out], temp_left, tt_cores, WD * 2, 0, 0, tt_shapes[0] * tt_shapes[1], tt_shapes[2] * tt_ranks[3], tt_ranks[2]);
    contract(temp_left, grad_cores, tt_cores, WD * 1, 0, 0, tt_shapes[0], tt_shapes[1] * tt_ranks[2], tt_ranks[1]);//G1


}
