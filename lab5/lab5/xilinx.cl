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
    layer: for(int i = 0; i < kNum; i++){
      out_x: for(int w = 0; w < kOutImSize; w++){
        out_y: for(int h = 0; h < kOutImSize; h++){
            float res00_00, res01_00, res10_00, res11_00;
            res00_00 = res01_00 = res10_00 = res11_00 = bias[i];
            // Convolution
            int weight_layer_position = i * kNum * kKernel * kKernel;
            int input_layer_size = kInImSize*kInImSize;
            int x_position_0 = (w * 2) * kInImSize;
            int y_position_0 = (h * 2);
            // __attribute__((xcl_pipeline_loop))  
            convolutions: for (int j = 0; j < kNum; ++j) {
              weight_x: for (int p = 0; p < kKernel; ++p) {
                weight_y: for (int q = 0; q < kKernel; ++q) {
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
}
