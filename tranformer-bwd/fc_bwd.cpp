//#include "fc_bwd.h"
//
//void compute_tt_grad(float* tt_cores, int* tt_ranks, int* tt_shapes, int input_dims, int output_dims, float* input, float* output, float* grad_output, float* grad_input, float* grad_cores) {
//    /*
//
//    tt_cores: weights represented by TT
//    tt_ranks: [r_0,r_1,...,r_d+1]
//    tt_shapes: [i_1, i_2,...i_m, j_1,j_2,...,j_n], n + m =d
//    input: a vector with shape i_1*i_2*...*i_m
//    output: a vector with shape j_1*j_2*...*j_n
//    grad_output: a vector calculated by last layer with shape j_1*j_2*...*j_n
//    grad_input: a vector calculated by Wdy
//    grad_cores: a arrary stores the result for the gradient to each core
//
//    */
//    TYPE_WEIGHT left_cores[num_cores-1][1000];
//    TYPE_WEIGHT right_cores[num_cores-1][1000];
//
//    // store outer product of X*Y
//    TYPE_DATA outer_product[5000];
//    int outer_dim = input_dims * output_dims;
//    for (int i = 0; i < input_dims; i++) {
//        for (int j = 0; j < output_dims; j++) {
//            outer_product[i * output_dims + j] = input[i] * grad_output[j];
//        }
//    }
//
//    // flatten left cores w.r.t i-th core
//    for (int i = 0; i < tt_ranks[0] * tt_shapes[0] * tt_ranks[1]; i++) {
//        left_cores[0][i] = tt_cores[i];
//    }
//    int shapeA = 1;
//    int left_shape[num_cores];
//    left_shape[0] = 1;
//    for (int i = 1; i < num_cores - 1; i++) {
//        shapeA *= tt_shapes[i - 1];
//        left_shape[i] = shapeA * tt_ranks[i];
//        contract(left_cores[i - 1], left_cores[i], tt_cores, WD * i, 0, 0, shapeA, tt_ranks[i], tt_shapes[i] * tt_ranks[i + 1]);
//    }
//    left_shape[num_cores - 1] = shapeA * tt_shapes[num_cores - 2] * tt_ranks[num_cores - 1];
//
//    //flatten right cores w.r.t i-th core
//    for (int i = 0; i < tt_ranks[num_cores-1] * tt_shapes[num_cores-1] * tt_ranks[num_cores]; i++) {
//        right_cores[0][i] = tt_cores[i + WD * (num_cores - 1)];
//    }
//    int shapeC = 1;
//    int right_shape[num_cores - 1];
//    right_shape[0] = 1;
//    for (int i = 1; i < num_cores - 1; i++) {
//        shapeC *= tt_shapes[num_cores - i];
//        right_shape[i] = shapeC * tt_ranks[num_cores - i];
//        contract(tt_cores, right_cores[i], right_cores[i - 1], 0, WD * (num_cores - i - 1), 0, tt_ranks[num_cores - i - 1] * tt_shapes[num_cores - i - 1], tt_ranks[num_cores - i], shapeC);
//    }
//    right_shape[num_cores - 1] = shapeC * tt_shapes[1] * tt_ranks[1];
//
//    TYPE_DATA temp[5000];
//    //calculate gradient to each core
//    for (int i = 0; i < num_cores; i++) {
//        if (i == 0) {
//            contract(right_cores[num_cores - 2], grad_cores, outer_product, 0, 0, 0, tt_ranks[1], outer_dim / tt_shapes[0], tt_shapes[0]);
//        }
//        else if (i == num_cores - 1) {
//            contract(left_cores[num_cores - 2], grad_cores, outer_product, 0, 0, WD * (num_cores - 1), tt_ranks[num_cores - 1], outer_dim / tt_shapes[num_cores - 1], tt_shapes[num_cores - 1]);
//        }
//        else {
//            //do outer_product for left_core and right_core
//            for (int j = 0; j < left_shape[i - 1]; j++) {
//                for (int k = 0; k < right_shape[num_cores - i - 2];k++) {
//                    temp[j * right_shape[num_cores - i - 2] + k] = left_cores[i - 1][j] * right_cores[num_cores - i - 2][k];
//                }
//            }
//            contract(temp, grad_cores, outer_product, 0, 0, WD * i, tt_ranks[i] * tt_ranks[i + 1], outer_dim / tt_shapes[i], tt_shapes[i]);
//        }
//    }
//
//    //calculate dL/dx = W* dL/dy
//    contract(left_cores[num_cores - 2], temp, tt_cores, WD * (num_cores - 1), 0, 0, shapeA * tt_shapes[num_cores - 2], tt_ranks[num_cores - 1], tt_shapes[num_cores - 1]);
//    contract(temp, grad_input, grad_output, 0, 0, 0, input_dims, output_dims, 1);
//
//}
