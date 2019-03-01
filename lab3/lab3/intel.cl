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
  int pixel_y = get_global_id(2);

  int layer_size = kOutImSize * kOutImSize;

  float res00_1, res01_1, res10_1, res11_1;
  res00_1 = res01_1 = res10_1 = res11_1 = bias[layer];

  float res00_2, res01_2, res10_2, res11_2;
  res00_2 = res01_2 = res10_2 = res11_2 = bias[layer];

  // Convolution
  int weight_layer_position = layer * kNum * kKernel * kKernel;
  int input_layer_size = kInImSize*kInImSize;
  int x_position_1 = (pixel_x * 2) * kInImSize;
  int x_position_2 = ((pixel_x + 1) * 2) * kInImSize;
  int y_position_1 = (pixel_y * 2);
  int y_position_2 = ((pixel_y + 1) * 2); 
  for (int j = 0; j < kNum; ++j) {
    for (int p = 0; p < kKernel; ++p) {
      for (int q = 0; q < kKernel; ++q) {
        float curr_weight =  weight[weight_layer_position + (p * kKernel) + q];
        res00_1 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_1 + q];
        res10_1 += curr_weight *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_1 + q];
        res01_1 += curr_weight *
                    input[x_position_1 + p * kInImSize + y_position_1 + 1 + q];
        res11_1 += weight[weight_layer_position + (p * kKernel) + q] *
                    input[x_position_1 + (1 + p) * kInImSize + y_position_1 + 1 + q];

        res00_2 += curr_weight *
                    input[x_position_2 + p * kInImSize + y_position_1 + q];
        res10_2 += curr_weight *
                    input[x_position_2 + (1 + p) * kInImSize + y_position_1 + q];
        res01_2 += curr_weight *
                    input[x_position_2 + p * kInImSize + y_position_1 + 1 + q];
        res11_2 += curr_weight *
                    input[x_position_2 + (1 + p) * kInImSize + y_position_1 + 1 + q];
      }
    }
    weight_layer_position += kKernel * kKernel;
    x_position_1 += input_layer_size;
    x_position_2 += input_layer_size;
  }
  //avoid function calls
  float max_val_1 = (res00_1 > res01_1 ? res00_1 : res01_1) > (res10_1 > res11_1 ? res10_1 : res11_1) ? (res00_1 > res01_1 ? res00_1 : res01_1) : (res10_1 > res11_1 ? res10_1 : res11_1);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y] = max_val_1 > 0 ? max_val_1 : 0;

  float max_val_2 = (res00_2 > res01_2 ? res00_2 : res01_2) > (res10_2 > res11_2 ? res10_2 : res11_2) ? (res00_2 > res01_2 ? res00_2 : res01_2) : (res10_2 > res11_2 ? res10_2 : res11_2);
  output[(layer * layer_size) + ((pixel_x + 1) * kOutImSize) + pixel_y] = max_val_2 > 0 ? max_val_2 : 0;

}
