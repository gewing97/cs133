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

    layer: for (int i = 0; i < kNum; i++){
        out_x: for (int w = 0; w < kOutImSize / 2; w++){
            out_y: for (int h = 0; h < kOutImSize / 4; h++){
                int pixel_x = w * 2;
                int pixel_y = h * 4;

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
                int weight_layer_position = i * kNum * kKernel * kKernel;
                int input_layer_size = kInImSize*kInImSize;
                int x_positions[] = { (pixel_x * 2) * kInImSize, ((pixel_x + 1) * 2) * kInImSize;}
                int x_position_0 = (pixel_x * 2) * kInImSize;
                int x_position_1 = ((pixel_x + 1) * 2) * kInImSize;
                int y_position_0 = (pixel_y * 2);
                int y_position_1 = ((pixel_y + 1) * 2);
                int y_position_2 = ((pixel_y + 2) * 2); 
                int y_position_3 = ((pixel_y + 3) * 2); 

                __attribute__((xcl_pipeline_loop))
                convolution: for (int j = 0; j < kNum; ++j) {
                    float temp00_00, temp01_00, temp10_00, temp11_00;
                    temp00_00 = temp01_00 = temp10_00 = temp11_00 = 0;

                    float temp00_01, temp01_01, temp10_01, temp11_01;
                    temp00_01 = temp01_01 = temp10_01 = temp11_01 = 0;

                    float temp00_10, temp01_10, temp10_10, temp11_10;
                    temp00_10 = temp01_10 = temp10_10 = temp11_10 = 0;

                    float temp00_11, temp01_11, temp10_11, temp11_11;
                    temp00_11 = temp01_11 = temp10_11 = temp11_11 = 0;

                    float temp00_02, temp01_02, temp10_02, temp11_02;
                    temp00_02 = temp01_02 = temp10_02 = temp11_02 = 0;

                    float temp00_03, temp01_03, temp10_03, temp11_03;
                    temp00_03 = temp01_03 = temp10_03 = temp11_03 = 0;

                    float temp00_12, temp01_12, temp10_12, temp11_12;
                    temp00_12 = temp01_12 = temp10_12 = temp11_12 = 0;

                    float temp00_13, temp01_13, temp10_13, temp11_13;
                    temp00_13 = temp01_13 = temp10_13 = temp11_13 = 0;
                    weight_x: for (int p = 0; p < kKernel; ++p) {
                        weight_y: for (int q = 0; q < kKernel; ++q) {
                            float curr_weight =  weight[weight_layer_position + (p * kKernel) + q];
                            temp00_00 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_0 + q];
                            temp10_00 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_0 + q];
                            temp01_00 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_0 + 1 + q];
                            temp11_00 += weight[weight_layer_position + (p * kKernel) + q] *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_0 + 1 + q];

                            temp00_01 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_1 + q];
                            temp10_01 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_1 + q];
                            temp01_01 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_1 + 1 + q];
                            temp11_01 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_1 + 1 + q];

                            temp00_10 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_0 + q];
                            temp10_10 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_0 + q];
                            temp01_10 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_0 + 1 + q];
                            temp11_10 += weight[weight_layer_position + (p * kKernel) + q] *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_0 + 1 + q];

                            temp00_11 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_1 + q];
                            temp10_11 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_1 + q];
                            temp01_11 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_1 + 1 + q];
                            temp11_11 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_1 + 1 + q];

                            temp00_02 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_2 + q];
                            temp10_02 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_2 + q];
                            temp01_02 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_2 + 1 + q];
                            temp11_02 += weight[weight_layer_position + (p * kKernel) + q] *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_2 + 1 + q];

                            temp00_03 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_3 + q];
                            temp10_03 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_3 + q];
                            temp01_03 += curr_weight *
                                        input[x_position_0 + p * kInImSize + y_position_3 + 1 + q];
                            temp11_03 += curr_weight *
                                        input[x_position_0 + (1 + p) * kInImSize + y_position_3 + 1 + q];

                            temp00_12 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_2 + q];
                            temp10_12 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_2 + q];
                            temp01_12 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_2 + 1 + q];
                            temp11_12 += weight[weight_layer_position + (p * kKernel) + q] *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_2 + 1 + q];

                            temp00_13 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_3 + q];
                            temp10_13 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_3 + q];
                            temp01_13 += curr_weight *
                                        input[x_position_1 + p * kInImSize + y_position_3 + 1 + q];
                            temp11_13 += curr_weight *
                                        input[x_position_1 + (1 + p) * kInImSize + y_position_3 + 1 + q];
                        }
                    }
                    weight_layer_position += kKernel * kKernel;
                    x_position_0 += input_layer_size;
                    x_position_1 += input_layer_size;
                    res00_00 += temp00_00;
                    res01_00 += temp01_00;
                    res10_00 += temp10_00;
                    res11_00 += temp11_00;

                    res00_01 += temp00_01;
                    res01_01 += temp01_01;
                    res10_01 += temp10_01;
                    res11_01 += temp11_01;

                    res00_10 += temp00_10;
                    res01_10 += temp01_10;
                    res10_10 += temp10_10;
                    res11_10 += temp11_10;

                    res00_11 += temp00_11;
                    res01_11 += temp01_11;
                    res10_11 += temp10_11;
                    res11_11 += temp11_11;

                    res00_02 += temp00_02;
                    res01_02 += temp01_02;
                    res10_02 += temp10_02;
                    res11_02 += temp11_02;

                    res00_12 += temp00_12;
                    res01_12 += temp01_12;
                    res10_12 += temp10_12;
                    res11_12 += temp11_12;

                    res00_03 += temp00_03;
                    res01_03 += temp01_03;
                    res10_03 += temp10_03;
                    res11_03 += temp11_03;

                    res00_13 += temp00_13;
                    res01_13 += temp01_13;
                    res10_13 += temp10_13;
                    res11_13 += temp11_13;
                }
                //avoid function calls
                float max_val_00 = (res00_00 > res01_00 ? res00_00 : res01_00) > (res10_00 > res11_00 ? res10_00 : res11_00) ? (res00_00 > res01_00 ? res00_00 : res01_00) : (res10_00 > res11_00 ? res10_00 : res11_00);
                output[(i* layer_size) + (pixel_x * kOutImSize) + pixel_y] = max_val_00 > 0 ? max_val_00 : 0;

                float max_val_01 = (res00_01 > res01_01 ? res00_01 : res01_01) > (res10_01 > res11_01 ? res10_01 : res11_01) ? (res00_01 > res01_01 ? res00_01 : res01_01) : (res10_01 > res11_01 ? res10_01 : res11_01);
                output[(i* layer_size) + (pixel_x * kOutImSize) + pixel_y + 1] = max_val_01 > 0 ? max_val_01 : 0;

                float max_val_10 = (res00_10 > res01_10 ? res00_10 : res01_10) > (res10_10 > res11_10 ? res10_10 : res11_10) ? (res00_10 > res01_10 ? res00_10 : res01_10) : (res10_10 > res11_10 ? res10_10 : res11_10);
                output[(i* layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y] = max_val_10 > 0 ? max_val_10 : 0;

                float max_val_11 = (res00_11 > res01_11 ? res00_11 : res01_11) > (res10_11 > res11_11 ? res10_11 : res11_11) ? (res00_11 > res01_11 ? res00_11 : res01_11) : (res10_11 > res11_11 ? res10_11 : res11_11);
                output[(i* layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 1] = max_val_11 > 0 ? max_val_11 : 0;


                float max_val_02 = (res00_02 > res01_02 ? res00_02 : res01_02) > (res10_02 > res11_02 ? res10_02 : res11_02) ? (res00_02 > res01_02 ? res00_02 : res01_02) : (res10_02 > res11_02 ? res10_02 : res11_02);
                output[(i* layer_size) + (pixel_x * kOutImSize) + pixel_y + 2] = max_val_02 > 0 ? max_val_02 : 0;

                float max_val_03 = (res00_03 > res01_03 ? res00_03 : res01_03) > (res10_03 > res11_03 ? res10_03 : res11_03) ? (res00_03 > res01_03 ? res00_03 : res01_03) : (res10_03 > res11_03 ? res10_03 : res11_03);
                output[(i* layer_size) + (pixel_x * kOutImSize) + pixel_y + 3] = max_val_03 > 0 ? max_val_03 : 0;

                float max_val_12 = (res00_12 > res01_12 ? res00_12 : res01_12) > (res10_12 > res11_12 ? res10_12 : res11_12) ? (res00_12 > res01_12 ? res00_12 : res01_12) : (res10_12 > res11_12 ? res10_12 : res11_12);
                output[(i* layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 2] = max_val_12 > 0 ? max_val_12 : 0;

                float max_val_13 = (res00_13 > res01_13 ? res00_13 : res01_13) > (res10_13 > res11_13 ? res10_13 : res11_13) ? (res00_13 > res01_13 ? res00_13 : res01_13) : (res10_13 > res11_13 ? res10_13 : res11_13);
                output[(i* layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 3] = max_val_13 > 0 ? max_val_13 : 0;
            }
        }
    }        
}
