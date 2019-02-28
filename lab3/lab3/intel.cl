__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__kernel
void CnnKernel(__global const float* input, __global const float* weight,
               __global const float* bias, __global float* output) {

  for (int i = 0; i < kImSize; ++i) {
    for (int h = 0; h < kImSize; ++h)
      output[i][h] = *bias;
  }

  // Convolution
  for (int j = 0; j < kNum; ++j) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        for (int p = 0; p < kKernel; ++p) {
          for (int q = 0; q < kKernel; ++q)
            output[h][w] += weight[j][p][q] * input[j][h + p][w + q];
        }
      }
    }
  }

  // ReLU
  for (int h = 0; h < kImSize; ++h) {
    for (int w = 0; w < kImSize; ++w) {
      output[h][w] = max(0.f, output[h][w]);
    }
  }

  // Max pooling
  for (int h = 0; h < kOutImSize; ++h) {
    for (int w = 0; w < kOutImSize; ++w) {
      output[h][w] = max(
          max(output[h * 2][w * 2    ], output[h * 2 + 1][w * 2    ]),
          max(output[h * 2][w * 2 + 1], output[h * 2 + 1][w * 2 + 1]));
    }
  }
}
