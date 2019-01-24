// Header inclusions, if any...

#include "gemm.h"
#include <vector>
#include <cstring>

// Using declarations, if any...


void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],  float c[kI][kJ]) {
    #pragma omp parallel for 
    for (int i = 0; i < kI; ++i) {
        std::memset(c[i], 0, sizeof(float) * kJ);
    }
    int const vert_block_size = 8;
    int const horz_block_size = 8;
    #pragma omp parallel for
    for(int vertical = 0; vertical < kI; vertical += vert_block_size){
        for(int horizontal = 0; horizontal < kJ; horizontal += horz_block_size){
            int vert_limit = std::min(vertical + vert_block_size, kI);
            for(int i = 0; i < vert_limit; i++){
                int horz_limit = std::min(horizontal + horz_block_size, kJ);
                for(int k = 0; k < vert_limit; k++){
                    for(int j = 0; j < horz_limit; j++){
                        c[i][j] = a[i][k] * b[k][j];
                    }
                }
            }
        }
   }
}
