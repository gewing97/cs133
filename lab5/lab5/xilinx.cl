__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define input(j, h, w) \
   input[((j) * kInImSize * kInImSize + (h) * kInImSize + (w))]
#define weight(j, h, w, q) \
    weight[(j * kNum * kKernel * kKernel) + (h * kKernel * kKernel) + (w * kKernel) + q]
#define bias(j) \
    bias[j]
#define output(j, h, w) \
    output[(j * kOutImSize * kOutImSize) + (h * kOutImSize) + w]

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {

    int layer_size = kOutImSize * kOutImSize;
    
    // __attribute__((xcl_pipeline_loop)) complained about unrolling x
    layer: for (int i = 0; i < kNum; i++){
        out_x: for (int w = 0; w < kOutImSize; w+= 2){
            out_y: for (int h = 0; h < kOutImSize; h+=4){
                float res00_00, res01_00, res10_00, res11_00;
                res00_00 = res01_00 = res10_00 = res11_00 = bias[i];

                float res00_01, res01_01, res10_01, res11_01;
                res00_01 = res01_01 = res10_01 = res11_01 = bias[i];

                float res00_10, res01_10, res10_10, res11_10;
                res00_10 = res01_10 = res10_10 = res11_10 = bias[i];

                float res00_11, res01_11, res10_11, res11_11;
                res00_11 = res01_11 = res10_11 = res11_11 = bias[i];

                float res00_02, res01_02, res10_02, res11_02;
                res00_02 = res01_02 = res10_02 = res11_02 = bias[i];

                float res00_03, res01_03, res10_03, res11_03;
                res00_03 = res01_03 = res10_03 = res11_03 = bias[i];

                float res00_12, res01_12, res10_12, res11_12;
                res00_12 = res01_12 = res10_12 = res11_12 = bias[i];

                float res00_13, res01_13, res10_13, res11_13;
                res00_13 = res01_13 = res10_13 = res11_13 = bias[i];

                // Convolution
                // int weight_layer_position = i * kNum * kKernel * kKernel;
                int input_layer_size = kInImSize*kInImSize;
                int x_position_0 = w * kInImSize;
                int x_position_1 = (w + 1) * kInImSize;
                int y_position_0 = h;
                int y_position_1 = h + 1;
                int y_position_2 = h + 2; 
                int y_position_3 = h + 3;  
                // __attribute__((xcl_pipeline_loop))  
                convolutions: for (int j = 0; j < kNum; ++j) {
                    // float curr_weights[kKernel][kKernel]
                    // __attribute__((xcl_array_partition(complete, 1)))
                    // __attribute__((xcl_array_partition(complete, 2)));
                    // __attribute__((xcl_pipeline_loop))
                    // load_weights: for (int p = 0; p < kKernel; p++){
                    //     for (int q = 0; q < kKernel; q++){
                    //         curr_weights[p][q] = weight(i, j, p, q);
                    //     }
                    // }
                    weight_x: for (int p = 0; p < kKernel; ++p) {
                        weight_y: for (int q = 0; q < kKernel; ++q) {
                            float curr_weight = weight(i, j, p, q);// weight[weight_layer_position + (p * kKernel) + q];
                            res00_00 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_0 + q];
                            res10_00 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_0 + q];
                            res01_00 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_0 + 1 + q];
                            res11_00 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_0 + 1 + q];

                            res00_01 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_1 + q];
                            res10_01 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_1 + q];
                            res01_01 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_1 + 1 + q];
                            res11_01 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_1 + 1 + q];

                            res00_10 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_0 + q];
                            res10_10 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_0 + q];
                            res01_10 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_0 + 1 + q];
                            res11_10 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_0 + 1 + q];

                            res00_11 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_1 + q];
                            res10_11 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_1 + q];
                            res01_11 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_1 + 1 + q];
                            res11_11 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_1 + 1 + q];

                            res00_02 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_2 + q];
                            res10_02 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_2 + q];
                            res01_02 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_2 + 1 + q];
                            res11_02 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_2 + 1 + q];

                            res00_03 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_3 + q];
                            res10_03 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_3 + q];
                            res01_03 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_3 + 1 + q];
                            res11_03 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_3 + 1 + q];

                            res00_12 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_2 + q];
                            res10_12 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_2 + q];
                            res01_12 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_2 + 1 + q];
                            res11_12 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_2 + 1 + q];

                            res00_13 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_3 + q];
                            res10_13 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_3 + q];
                            res01_13 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_3 + 1 + q];
                            res11_13 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_3 + 1 + q];
                        }
                    }
                    // weight_layer_position += kKernel * kKernel;
                    x_position_0 += input_layer_size;
                    x_position_1 += input_layer_size;
                }
                //avoid function calls
                float max_val_00 = (res00_00 > res01_00 ? res00_00 : res01_00) > (res10_00 > res11_00 ? res10_00 : res11_00) ? (res00_00 > res01_00 ? res00_00 : res01_00) : (res10_00 > res11_00 ? res10_00 : res11_00);
                output[(i * layer_size) + (w * kOutImSize) + h] = max_val_00 > 0 ? max_val_00 : 0;

                float max_val_01 = (res00_01 > res01_01 ? res00_01 : res01_01) > (res10_01 > res11_01 ? res10_01 : res11_01) ? (res00_01 > res01_01 ? res00_01 : res01_01) : (res10_01 > res11_01 ? res10_01 : res11_01);
                output[(i * layer_size) + (w * kOutImSize) + h + 1] = max_val_01 > 0 ? max_val_01 : 0;

                float max_val_10 = (res00_10 > res01_10 ? res00_10 : res01_10) > (res10_10 > res11_10 ? res10_10 : res11_10) ? (res00_10 > res01_10 ? res00_10 : res01_10) : (res10_10 > res11_10 ? res10_10 : res11_10);
                output[(i * layer_size) + ((w + 1) * kOutImSize) + h] = max_val_10 > 0 ? max_val_10 : 0;

                float max_val_11 = (res00_11 > res01_11 ? res00_11 : res01_11) > (res10_11 > res11_11 ? res10_11 : res11_11) ? (res00_11 > res01_11 ? res00_11 : res01_11) : (res10_11 > res11_11 ? res10_11 : res11_11);
                output[(i * layer_size) + ((w + 1) * kOutImSize) + h + 1] = max_val_11 > 0 ? max_val_11 : 0;


                float max_val_02 = (res00_02 > res01_02 ? res00_02 : res01_02) > (res10_02 > res11_02 ? res10_02 : res11_02) ? (res00_02 > res01_02 ? res00_02 : res01_02) : (res10_02 > res11_02 ? res10_02 : res11_02);
                output[(i * layer_size) + (w * kOutImSize) + h + 2] = max_val_02 > 0 ? max_val_02 : 0;

                float max_val_03 = (res00_03 > res01_03 ? res00_03 : res01_03) > (res10_03 > res11_03 ? res10_03 : res11_03) ? (res00_03 > res01_03 ? res00_03 : res01_03) : (res10_03 > res11_03 ? res10_03 : res11_03);
                output[(i * layer_size) + (w * kOutImSize) + h + 3] = max_val_03 > 0 ? max_val_03 : 0;

                float max_val_12 = (res00_12 > res01_12 ? res00_12 : res01_12) > (res10_12 > res11_12 ? res10_12 : res11_12) ? (res00_12 > res01_12 ? res00_12 : res01_12) : (res10_12 > res11_12 ? res10_12 : res11_12);
                output[(i * layer_size) + ((w + 1) * kOutImSize) + h + 2] = max_val_12 > 0 ? max_val_12 : 0;

                float max_val_13 = (res00_13 > res01_13 ? res00_13 : res01_13) > (res10_13 > res11_13 ? res10_13 : res11_13) ? (res00_13 > res01_13 ? res00_13 : res01_13) : (res10_13 > res11_13 ? res10_13 : res11_13);
                output[(i * layer_size) + ((w + 1) * kOutImSize) + h + 3] = max_val_13 > 0 ? max_val_13 : 0;
        }
      }
    }
}
