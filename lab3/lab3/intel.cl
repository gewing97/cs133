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
  int pixel_x = get_global_id(1);
  int pixel_y = get_global_id(2);

  int layer_size = kOutImSize * kOutImSize;

  float res00, res01, res10, res11;
  res00 = res01 = res10 = res11 = bias[layer];

  // Convolution
  int weight_layer_position = layer * kNum * kKernel * kKernel;
  int input_layer_size = kInImSize*kInImSize;
  for (int j = 0; j < kNum; ++j) {
    int input_layer_position = (j * input_layer_size);
    for (int p = 0; p < kKernel; ++p) {
      int x_position = (pixel_x * 2) * kInImSize;
      for (int q = 0; q < kKernel; ++q) {
        int y_position = (pixel_y * 2);
        res00 += weight[weight_layer_position + (j * kKernel * kKernel) + (p * kKernel) + q] *
                    input[input_layer_position + x_position + p * kInImSize + y_position + q];
        res10 += weight[weight_layer_position + (j * kKernel * kKernel) + (p * kKernel) + q] *
                    input[input_layer_position + x_position + (1 + p) * kInImSize + y_position + q];
        res01 += weight[weight_layer_position + (j * kKernel * kKernel) + (p * kKernel) + q] *
                    input[input_layer_position + x_position + p * kInImSize + y_position + 1 + q];
        res11 += weight[weight_layer_position + (j * kKernel * kKernel) + (p * kKernel) + q] *
                    input[input_layer_position + x_position + (1 + p) * kInImSize + y_position + 1 + q];
      }
    }
  }
  //avoid function calls
  float max_val = (res00 > res01 ? res00 : res01) > (res10 > res11 ? res10 : res11) ? (res00 > res01 ? res00 : res01) : (res10 > res11 ? res10 : res11);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y] = max_val > 0 ? max_val : 0;
}
