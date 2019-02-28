__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__kernel
void CnnKernel(__global const float* input, __global const float* weight,
               __global const float* bias, __global float* output) {

    
  for (int i = 0; i < kImSize; ++i)
    output[i] = *bias;
  printf("input %d\n", input[0]);
  printf("weight %d\n", weight[0]);
  printf("bias %d\n", bias);
  printf("output %d\n", output[0]);
}
