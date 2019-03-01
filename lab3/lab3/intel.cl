__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__kernel
void CnnKernel(__global const float* input, __global const float* weight,
               __global const float* bias, __global float* output) {
  // float C[2][2];


  int layer_size = kOutImSize * kOutImSize;

  int num_layers = kNum / get_global_size(0);
  int x_pixels = kOutImSize / get_global_size(1);
  int y_pixels = kOutImSize / get_global_size(2);

  if(((num_layers ^ 1) | (x_pixels ^ 2) | (y_pixels ^ 4))){
    int layer = get_global_id(0) * num_layers;
    int pixel_x = get_global_id(1) * x_pixels;
    int pixel_y = get_global_id(2) * y_pixels;
    for(int i = layer; i < layer + num_layers; i++){
      for(int w = pixel_x; w < pixel_x + x_pixels; w++){
        for(int h = pixel_y; h < pixel_y + y_pixels; h++){
            // printf("indices %d %d %d\b", i, w, h);

            float res00_00, res01_00, res10_00, res11_00;
            res00_00 = res01_00 = res10_00 = res11_00 = bias[i];
            // Convolution
            int weight_layer_position = i * kNum * kKernel * kKernel;
            int input_layer_size = kInImSize*kInImSize;
            int x_position_0 = (w * 2) * kInImSize;
            int y_position_0 = (h * 2);
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
                  res11_00 += weight[weight_layer_position + (p * kKernel) + q] *
                              input[x_position_0 + (1 + p) * kInImSize + y_position_0 + 1 + q];
                }
              }
              weight_layer_position += kKernel * kKernel;
              x_position_0 += input_layer_size;
            }
            //avoid function calls
            float max_val_00 = (res00_00 > res01_00 ? res00_00 : res01_00) > (res10_00 > res11_00 ? res10_00 : res11_00) ? (res00_00 > res01_00 ? res00_00 : res01_00) : (res10_00 > res11_00 ? res10_00 : res11_00);
            output[(i * layer_size) + (w * kOutImSize) + h] = max_val_00 > 0 ? max_val_00 : 0;
        }
      }
    }
    return;
  }
  // most optimized version of my code
  int layer = get_global_id(0);
  int pixel_x = get_global_id(1) * 2;
  int pixel_y = get_global_id(2) * 4;
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

  // Convolution
  int weight_layer_position = layer * kNum * kKernel * kKernel;
  int input_layer_size = kInImSize*kInImSize;
  int x_position_0 = (pixel_x * 2) * kInImSize;
  int x_position_1 = ((pixel_x + 1) * 2) * kInImSize;
  int y_position_0 = (pixel_y * 2);
  int y_position_1 = ((pixel_y + 1) * 2);
  int y_position_2 = ((pixel_y + 2) * 2); 
  int y_position_3 = ((pixel_y + 3) * 2);  
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
        res11_00 += weight[weight_layer_position + (p * kKernel) + q] *
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
        res11_10 += weight[weight_layer_position + (p * kKernel) + q] *
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
        res11_02 += weight[weight_layer_position + (p * kKernel) + q] *
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
        res11_12 += weight[weight_layer_position + (p * kKernel) + q] *
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
}

