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

                    // Allocate memory on heap to avoid stack overflow.
  static float C[kNum][kImSize][kImSize];

  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w)
        C[i][h][w] = bias(i);
    }
  }

  // Convolution
  for (int i = 0; i < kNum; ++i) {
    for (int j = 0; j < kNum; ++j) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q)
              C[i][h][w] += weight(i,j,p,q) * input(j,h + p,w + q);
          }
        }
      }
    }
  }

  // ReLU
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C[i][h][w] = max(0.f, C[i][h][w]);
      }
    }
  }

  // Max pooling
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        output(i,h,w) = max(
            max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
            max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1]));
      }
    }
  }

    // for (int j = 0; j < kNum; j++){
    //     for (int h = 0; h < )
    //     float output_buf[kImSize][kImSize] // buffer of output
    //     ;
    //     float input_buf[kInImSize][kInImSize + kKernel - 1][kKernel] //buffer of input
    //     __attribute__((xcl_array_partition(cyclic, 8, 1)))  // cyclic partition factor of 8 in dim 1 of input_buf
    //     __attribute__((xcl_array_partition(complete, 3))) // complete partitioning for dim3 of input_buf
    //     ;

    //     float weight_buf[kKernel][kKernel] //buffer of weight
    //     __attribute__((xcl_array_partition(complete, 1))) // complete partitioning for dim 1 of weight_buf
    //     __attribute__((xcl_array_partition(complete, 2))) // complete partitioning for dim 2 of weight_buf
    //     ;

    //     //copy bias here
    //     // __attribute__((xcl_pipeline_loop))
    //     // for (int w = 0; w < kImSize; w++) {
    //     //     output_buf[h][w] = bias[h];
    //     // }

    //     //input load loop
    //     //load_in:
    //     __attribute__((xcl_pipeline_loop))
    //     for (int w = 0; w < kInImSize; w ++) {
    //         for (int q = 0; q < kKernel; ++q) { //make kKernel copy of input(j,h,w)
    //             input_buf[h][w - q + kKernel - 1][q] = input(j, h, w);
    //         }
    //     }

    //     //copy weight here
    //     // __attribute__((xcl_pipeline_loop))
    //     // for (int w = 0; w < kKernel; w++) {
    //     //     for (int q = 0; q < kKernel; q++){
    //     //         weight_buf[w][q] = weight(j, h, w, q);
    //     //     }
    //     // }

    //     //convolution loop
    //     //conv:
    //     __attribute__((xcl_pipeline_loop))
    //     for (int w = 0; w < kImSize; ++w) { //pipelined loop
    //         float tmp = 0; 
    //         for (int p = 0; p < kKernel; ++p) {  // unrolled loop
    //             for (int q = 0; q < kKernel; ++q) {  //unrolled loop
    //             tmp += //will be synthesized into tree reduction
    //                     weight_buf[p][q] *
    //                     input_buf[h + p][w + kKernel - 1][q];
    //             }
    //         }
    //         output_buf[h][w] += tmp; //store reduction result
    //     }

    //     // //copy output here          
    //     // __attribute__((xcl_pipeline_loop))
    //     // for (int i = 0; i < kOutImSize; i++) {
    //     //     for (int j = 0; j < kOutImSize; j++){
    //     //         out
    //     //     }
    //     // }
    // }
}
