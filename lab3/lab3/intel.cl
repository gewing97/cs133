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
  // int layer = gid / layer_size;
  // int pixel_x = (gid % layer_size) / kOutImSize;
  // int pixel_y = gid % kOutImSize;
  // for (int h = 0; h < 2; ++h) {
  //   for (int w = 0; w < 2; ++w)
  //     C[h][w] = bias[layer]; //which value of the bias do we want? based on gid
  // }


  float res00, res01, res10, res11;
  res00 = res01 = res10 = res11 = bias[layer];
  // Convolution
  int weight_layer_size = kNum * kKernel * kKernel;
  int input_layer_size = kInImSize*kInImSize;
  for (int j = 0; j < kNum; ++j) {
    // for (int h = 0; h < 2; ++h) {
      // for (int w = 0; w < 2; ++w) {
        for (int p = 0; p < kKernel; ++p) {
          for (int q = 0; q < kKernel; ++q) {
            res00 += weight[(layer * weight_layer_size) + (j * kKernel * kKernel) + (p * kKernel) + q] *
                        input[(j * input_layer_size) + (((pixel_x * 2) + 0 + p) * kInImSize) + ((pixel_y * 2) + 0 + q)];
            res10 += weight[(layer * weight_layer_size) + (j * kKernel * kKernel) + (p * kKernel) + q] *
                        input[(j * input_layer_size) + (((pixel_x * 2) + 1 + p) * kInImSize) + ((pixel_y * 2) + 0 + q)];
            res01 += weight[(layer * weight_layer_size) + (j * kKernel * kKernel) + (p * kKernel) + q] *
                        input[(j * input_layer_size) + (((pixel_x * 2) + 0 + p) * kInImSize) + ((pixel_y * 2) + 1 + q)];
            res11 += weight[(layer * weight_layer_size) + (j * kKernel * kKernel) + (p * kKernel) + q] *
                        input[(j * input_layer_size) + (((pixel_x * 2) + 1 + p) * kInImSize) + ((pixel_y * 2) + 1 + q)];
          }
        }
      // }
    // }
  }

  // // ReLU
  // for (int h = 0; h < 2; ++h) {
  //   for (int w = 0; w < 2; ++w) {
  //     // printf("value of c %f\n", C[h][w]);
  //     C[h][w] = max(0.f, C[h][w]);
  //   }
  // }

  //want to make it such that output is a single value
  //which means we only need 4 values of C
  // Max pooling
  // output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y] = max(
  //   max(C[0][0], C[1][0]),
  //   max(C[0][1], C[1][1]));

  //avoid function calls
  float max_val = (res00 > res01 ? res00 : res01) > (res10 > res11 ? res10 : res11) ? (res00 > res01 ? res00 : res01) : (res10 > res11 ? res10 : res11);
  output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y] = max_val > 0 ? max_val : 0;
  // printf("bias %f\n", bias[layer]);
  // printf("output %f at %d\n", output[(layer * layer_size) + (pixel_x * kOutImSize) + pixel_y], (layer * layer_size) + (pixel_x * kOutImSize) + pixel_y);
}
