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

    
    // int num_layers = kNum / get_global_size(0);
    // int x_pixels = kOutImSize / get_global_size(1);
    // int y_pixels = kOutImSize / get_global_size(2);
    // int layer = get_global_id(0) * num_layers;
    // int pixel_x = get_global_id(1) * x_pixels;
    // int pixel_y = get_global_id(2) * y_pixels;
    for(int i = 0; i < kNum; i++){
      for(int w = 0; w < kOutImSize; w++){
        for(int h = 0; h < kOutImSize; h++){
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

    // for (int i = 0; i < kNum; i++){
    //     for (int j = 0; j < kNum; j++){
    //         for (int h = 0; h < kImSize; h++){
    //             float output_buf[kImSize][kImSize] // buffer of output
    //             ;
    //             float input_buf[kInImSize][kInImSize + kKernel - 1][kKernel] //buffer of input
    //             __attribute__((xcl_array_partition(cyclic, 8, 1)))  // cyclic partition factor of 8 in dim 1 of input_buf
    //             __attribute__((xcl_array_partition(complete, 3))) // complete partitioning for dim3 of input_buf
    //             ;

    //             float weight_buf[kKernel][kKernel] //buffer of weight
    //             __attribute__((xcl_array_partition(complete, 1))) // complete partitioning for dim 1 of weight_buf
    //             __attribute__((xcl_array_partition(complete, 2))) // complete partitioning for dim 2 of weight_buf
    //             ;

    //             //copy bias here
    //             __attribute__((xcl_pipeline_loop))
    //             for (int w = 0; w < kImSize; w++) {
    //                 output_buf[h][w] = bias(i);
    //             }

    //             //input load loop
    //             //load_in:
    //             __attribute__((xcl_pipeline_loop))
    //             for (int w = 0; w < kInImSize; w ++) {
    //                 for (int q = 0; q < kKernel; ++q) { //make kKernel copy of input(j,h,w)
    //                     input_buf[h][w - q + kKernel - 1][q] = input(j, h, w);
    //                 }
    //             }

    //             //copy weight here
    //             // __attribute__((xcl_pipeline_loop))
    //             // for (int w = 0; w < kKernel; w++) {
    //             //     for (int q = 0; q < kKernel; q++){
    //             //         weight_buf[w][q] = weight(j, h, w, q);
    //             //     }
    //             // }

    //             //convolution loop
    //             //conv:
    //             __attribute__((xcl_pipeline_loop))
    //             for (int w = 0; w < kImSize; ++w) { //pipelined loop
    //                 float tmp = 0; 
    //                 for (int p = 0; p < kKernel; ++p) {  // unrolled loop
    //                     for (int q = 0; q < kKernel; ++q) {  //unrolled loop
    //                     tmp += //will be synthesized into tree reduction
    //                             weight_buf[p][q] *
    //                             input_buf[h + p][w + kKernel - 1][q];
    //                     }
    //                 }
    //                 output_buf[h][w] += tmp; //store reduction result
    //             }

    //             // //copy output here          
    //             // __attribute__((xcl_pipeline_loop))
    //             // for (int i = 0; i < kOutImSize; i++) {
    //             //     for (int j = 0; j < kOutImSize; j++){
    //             //         out
    //             //     }
    //             // }
    //         }
    //     }
    // }
}
