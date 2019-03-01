__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__kernel
void CnnKernel(__global const float* input, __global const float* weight,
               __global const float* bias, __global float* output) {
  // float C[2][2];

  int layer = get_global_id(0);
  int pixel_x = get_global_id(1) * 2;
  int pixel_y = get_global_id(2) * 4;

  int layer_size = kOutImSize * kOutImSize;

  float res00_00, res01_00, res10_00, res11_00;
  res00_00 = res01_00 = res10_00 = res11_00 = bias[layer];

  float res00_01, res01_01, res10_01, res11_01;
  res00_01 = res01_01 = res10_01 = res11_01 = bias[layer];

  float res00_10, res01_10, res10_10, res11_10;
  res00_10 = res01_10 = res10_10 = res11_10 = bias[layer];

  float res00_11, res01_11, res10_11, res11_11;
  res00_11 = res01_11 = res10_11 = res11_11 = bias[layer];

  float res00_02, res01_02, res10_02, res11_02;
  res00_02 = res01_02 = res10_02 = res11_02 = bias[layer];

  float res00_03, res01_03, res10_03, res11_03;
  res00_03 = res01_03 = res10_03 = res11_03 = bias[layer];

  float res00_12, res01_12, res10_12, res11_12;
  res00_12 = res01_12 = res10_12 = res11_12 = bias[layer];

  float res00_13, res01_13, res10_13, res11_13;
  res00_13 = res01_13 = res10_13 = res11_13 = bias[layer];

  float res00_04, res01_04, res10_04, res11_04;
  res00_04 = res01_04 = res10_04 = res11_04 = bias[layer];

  float res00_05, res01_05, res10_05, res11_05;
  res00_05 = res01_05 = res10_05 = res11_05 = bias[layer];

  float res00_14, res01_14, res10_14, res11_14;
  res00_14 = res01_14 = res10_14 = res11_14 = bias[layer];

  float res00_15, res01_15, res10_15, res11_15;
  res00_15 = res01_15 = res10_15 = res11_15 = bias[layer];

  float res00_06, res01_06, res10_06, res11_06;
  res00_06 = res01_06 = res10_06 = res11_06 = bias[layer];

  float res00_07, res01_07, res10_07, res11_07;
  res00_07 = res01_07 = res10_07 = res11_07 = bias[layer];

  float res00_16, res01_16, res10_16, res11_16;
  res00_16 = res01_16 = res10_16 = res11_16 = bias[layer];

  float res00_17, res01_17, res10_17, res11_17;
  res00_17 = res01_17 = res10_17 = res11_17 = bias[layer];

  // Convolution
  int weight_layer_position = layer * kNum * kKernel * kKernel;
  int input_layer_size = kInImSize*kInImSize;
  int x_position_0 = (pixel_x * 2) * kInImSize;
  int x_position_1 = ((pixel_x + 1) * 2) * kInImSize;
  int y_position_0 = (pixel_y * 2);
  int y_position_1 = ((pixel_y + 1) * 2);
  int y_position_2 = ((pixel_y + 2) * 2); 
  int y_position_3 = ((pixel_y + 3) * 2); 
  int y_position_4 = (pixel_y * 4);
  int y_position_5 = ((pixel_y + 5) * 2);
  int y_position_6 = ((pixel_y + 6) * 2); 
  int y_position_7 = ((pixel_y + 7) * 2);   
  for (int j = 0; j < kNum; ++j) {
    for (int p = 0; p < kKernel; ++p) {
      for (int q = 0; q < kKernel; ++q) {
        float curr_weight =  weight[weight_layer_position + (p * kKernel) + q];
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

        res00_04 += curr_weight *
                    input[x_position_0 + p * kInImSize + y_position_4 + q];
        res10_04 += curr_weight *
                    input[x_position_0 + (1 + p) * kInImSize + y_position_4 + q];
        res01_04 += curr_weight *
                    input[x_position_0 + p * kInImSize + y_position_4 + 1 + q];
        res11_04 += curr_weight *
                    input[x_position_0 + (1 + p) * kInImSize + y_position_4 + 1 + q];

        res00_05 += curr_weight *
                    input[x_position_0 + p * kInImSize + y_position_5 + q];
        res10_05 += curr_weight *
                    input[x_position_0 + (1 + p) * kInImSize + y_position_5 + q];
        res01_05 += curr_weight *
                    input[x_position_0 + p * kInImSize + y_position_5 + 1 + q];
        res11_05 += curr_weight *
                    input[x_position_0 + (1 + p) * kInImSize + y_position_5 + 1 + q];

        res00_14 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_4 + q];
        res10_14 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_4 + q];
        res01_14 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_4 + 1 + q];
        res11_14 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_4 + 1 + q];

        res00_15 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_5 + q];
        res10_15 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_5 + q];
        res01_15 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_5 + 1 + q];
        res11_15 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_5 + 1 + q];

        res00_06 += curr_weight *
                    input[x_position_0 + p * kInImSize + y_position_6 + q];
        res10_06 += curr_weight *
                    input[x_position_0 + (1 + p) * kInImSize + y_position_6 + q];
        res01_06 += curr_weight *
                    input[x_position_0 + p * kInImSize + y_position_6 + 1 + q];
        res11_06 += curr_weight *
                    input[x_position_0 + (1 + p) * kInImSize + y_position_6 + 1 + q];

        res00_07 += curr_weight *
                    input[x_position_0 + p * kInImSize + y_position_7 + q];
        res10_07 += curr_weight *
                    input[x_position_0 + (1 + p) * kInImSize + y_position_7 + q];
        res01_07 += curr_weight *
                    input[x_position_0 + p * kInImSize + y_position_7 + 1 + q];
        res11_07 += curr_weight *
                    input[x_position_0 + (1 + p) * kInImSize + y_position_7 + 1 + q];

        res00_16 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_6 + q];
        res10_16 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_6 + q];
        res01_16 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_6 + 1 + q];
        res11_16 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_6 + 1 + q];

        res00_17 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_7 + q];
        res10_17 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_7 + q];
        res01_17 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_7 + 1 + q];
        res11_17 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_7 + 1 + q];
      }
    }
    weight_layer_position += kKernel * kKernel;
    x_position_0 += input_layer_size;
    x_position_1 += input_layer_size;
  }
  //avoid function calls
  float max_val_00 = (res00_00 > res01_00 ? res00_00 : res01_00) > (res10_00 > res11_00 ? res10_00 : res11_00) ? (res00_00 > res01_00 ? res00_00 : res01_00) : (res10_00 > res11_00 ? res10_00 : res11_00);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y] = max_val_00 > 0 ? max_val_00 : 0;

  float max_val_01 = (res00_01 > res01_01 ? res00_01 : res01_01) > (res10_01 > res11_01 ? res10_01 : res11_01) ? (res00_01 > res01_01 ? res00_01 : res01_01) : (res10_01 > res11_01 ? res10_01 : res11_01);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y + 1] = max_val_01 > 0 ? max_val_01 : 0;

  float max_val_10 = (res00_10 > res01_10 ? res00_10 : res01_10) > (res10_10 > res11_10 ? res10_10 : res11_10) ? (res00_10 > res01_10 ? res00_10 : res01_10) : (res10_10 > res11_10 ? res10_10 : res11_10);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y] = max_val_10 > 0 ? max_val_10 : 0;

  float max_val_11 = (res00_11 > res01_11 ? res00_11 : res01_11) > (res10_11 > res11_11 ? res10_11 : res11_11) ? (res00_11 > res01_11 ? res00_11 : res01_11) : (res10_11 > res11_11 ? res10_11 : res11_11);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 1] = max_val_11 > 0 ? max_val_11 : 0;


  float max_val_02 = (res00_02 > res01_02 ? res00_02 : res01_02) > (res10_02 > res11_02 ? res10_02 : res11_02) ? (res00_02 > res01_02 ? res00_02 : res01_02) : (res10_02 > res11_02 ? res10_02 : res11_02);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y + 2] = max_val_02 > 0 ? max_val_02 : 0;

  float max_val_03 = (res00_03 > res01_03 ? res00_03 : res01_03) > (res10_03 > res11_03 ? res10_03 : res11_03) ? (res00_03 > res01_03 ? res00_03 : res01_03) : (res10_03 > res11_03 ? res10_03 : res11_03);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y + 3] = max_val_03 > 0 ? max_val_03 : 0;

  float max_val_12 = (res00_12 > res01_12 ? res00_12 : res01_12) > (res10_12 > res11_12 ? res10_12 : res11_12) ? (res00_12 > res01_12 ? res00_12 : res01_12) : (res10_12 > res11_12 ? res10_12 : res11_12);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 2] = max_val_12 > 0 ? max_val_12 : 0;

  float max_val_13 = (res00_13 > res01_13 ? res00_13 : res01_13) > (res10_13 > res11_13 ? res10_13 : res11_13) ? (res00_13 > res01_13 ? res00_13 : res01_13) : (res10_13 > res11_13 ? res10_13 : res11_13);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 3] = max_val_13 > 0 ? max_val_13 : 0;

  
  float max_val_04 = (res00_04 > res01_04 ? res00_04 : res01_04) > (res10_04 > res11_04 ? res10_04 : res11_04) ? (res00_04 > res01_04 ? res00_04 : res01_04) : (res10_04 > res11_04 ? res10_04 : res11_04);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y + 4] = max_val_04 > 0 ? max_val_04 : 0;

  float max_val_05 = (res00_05 > res01_05 ? res00_05 : res01_05) > (res10_05 > res11_05 ? res10_05 : res11_05) ? (res00_05 > res01_05 ? res00_05 : res01_05) : (res10_05 > res11_05 ? res10_05 : res11_05);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y + 5] = max_val_05 > 0 ? max_val_05 : 0;

  float max_val_14 = (res00_14 > res01_14 ? res00_14 : res01_14) > (res10_14 > res11_14 ? res10_14 : res11_14) ? (res00_14 > res01_14 ? res00_14 : res01_14) : (res10_14 > res11_14 ? res10_14 : res11_14);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 4] = max_val_14 > 0 ? max_val_14 : 0;

  float max_val_15 = (res00_15 > res01_15 ? res00_15 : res01_15) > (res10_15 > res11_15 ? res10_15 : res11_15) ? (res00_15 > res01_15 ? res00_15 : res01_15) : (res10_15 > res11_15 ? res10_15 : res11_15);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 5] = max_val_15 > 0 ? max_val_15 : 0;


  float max_val_06 = (res00_06 > res01_06 ? res00_06 : res01_06) > (res10_06 > res11_06 ? res10_06 : res11_06) ? (res00_06 > res01_06 ? res00_06 : res01_06) : (res10_06 > res11_06 ? res10_06 : res11_06);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y + 6] = max_val_06 > 0 ? max_val_06 : 0;

  float max_val_07 = (res00_07 > res01_07 ? res00_07 : res01_07) > (res10_07 > res11_07 ? res10_07 : res11_07) ? (res00_07 > res01_07 ? res00_07 : res01_07) : (res10_07 > res11_07 ? res10_07 : res11_07);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y + 7] = max_val_07 > 0 ? max_val_07 : 0;

  float max_val_16 = (res00_16 > res01_16 ? res00_16 : res01_16) > (res10_16 > res11_16 ? res10_16 : res11_16) ? (res00_16 > res01_16 ? res00_16 : res01_16) : (res10_16 > res11_16 ? res10_16 : res11_16);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 6] = max_val_16 > 0 ? max_val_16 : 0;

  float max_val_17 = (res00_17 > res01_17 ? res00_17 : res01_17) > (res10_17 > res11_17 ? res10_17 : res11_17) ? (res00_17 > res01_17 ? res00_17 : res01_17) : (res10_17 > res11_17 ? res10_17 : res11_17);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y + 7] = max_val_17 > 0 ? max_val_17 : 0;

}
