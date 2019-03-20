__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define input(j, h, w) \
   input[((j) * kInImSize * kInImSize + (h) * kInImSize + (w))]
#define weight(j, h, w, q) \
    weight[((j * kNum * kKernel * kKernel) + (h * kKernel * kKernel) + (w * kKernel) + q)]
#define bias(j) \
    bias[j]
#define output(j, h, w) \
    output[((j * kOutImSize * kOutImSize) + (h * kOutImSize) + w)]

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {

    for (int i = 0; i < kNum; ++i) {
        float C[kImSize][kImSize];
        for (int h = 0; h < kImSize; ++h) {
            for (int w = 0; w < kImSize; ++w)
                C[h][w] = bias[i];
            }

    // Convolution
        for (int j = 0; j < kNum; ++j) {
            float local_weight[kKernel][kKernel];
            __attribute__((xcl_pipeline_loop))
            for (int p = 0; p < kKernel; p++){
                for (int q =0; q < kKernel; q++){
                    local_weight[p][q] = weight(i,j,p,q);
                }
            }
            for (int h = 0; h < kImSize; ++h) {
                __attribute__((xcl_pipeline_loop))
                for (int w = 0; w < kImSize; ++w) {
                    float temp = 0;
                    for (int p = 0; p < kKernel; ++p) {
                        for (int q = 0; q < kKernel; ++q)
                            temp += local_weight[p][q] * input(j,h + p,w + q);
                    }
                    C[h][w] += temp;
                }
            }
        }

        for (int h = 0; h < kImSize; ++h) {
            for (int w = 0; w < kImSize; ++w) {
                C[h][w] = C[h][w] < 0.f ? 0 : C[h][w];
            }   
        }

    // Max pooling
        for (int h = 0; h < kOutImSize; ++h) {
            for (int w = 0; w < kOutImSize; ++w) {
                output(i,h,w) = max(
                    max(C[h * 2][w * 2    ], C[h * 2 + 1][w * 2    ]),
                    max(C[h * 2][w * 2 + 1], C[h * 2 + 1][w * 2 + 1]));
            }
        }
    }    
}
