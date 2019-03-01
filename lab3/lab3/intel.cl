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
  int weight_layer_size = kNum * kKernel * kKernel;
  int input_layer_size = kInImSize*kInImSize;
  for (int j = 0; j < kNum; ++j) {
    for (int p = 0; p < kKernel; ++p) {
      for (int q = 0; q < kKernel; ++q) {
        res00 += weight[(layer * weight_layer_size) + (j * kKernel * kKernel) + (p * kKernel) + q] *
                    input[(j * input_layer_size) + (((pixel_x << 1) + 0 + p) * kInImSize) + ((pixel_y << 1) + 0 + q)];
        res10 += weight[(layer * weight_layer_size) + (j * kKernel * kKernel) + (p * kKernel) + q] *
                    input[(j * input_layer_size) + (((pixel_x << 1) + 1 + p) * kInImSize) + ((pixel_y << 1) + 0 + q)];
        res01 += weight[(layer * weight_layer_size) + (j * kKernel * kKernel) + (p * kKernel) + q] *
                    input[(j * input_layer_size) + (((pixel_x << 1) + 0 + p) * kInImSize) + ((pixel_y << 1) + 1 + q)];
        res11 += weight[(layer * weight_layer_size) + (j * kKernel * kKernel) + (p * kKernel) + q] *
                    input[(j * input_layer_size) + (((pixel_x << 1) + 1 + p) * kInImSize) + ((pixel_y << 1) + 1 + q)];
      }
    }
  }
  //avoid function calls
  float max_val = (res00 > res01 ? res00 : res01) > (res10 > res11 ? res10 : res11) ? (res00 > res01 ? res00 : res01) : (res10 > res11 ? res10 : res11);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y] = max_val > 0 ? max_val : 0;
}
