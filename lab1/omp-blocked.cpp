// Header inclusions, if any...

#include "gemm.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <omp.h>

// Using declarations, if any...



void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],  float c[kI][kJ]) {
    #pragma omp parallel for 
    for (int i = 0; i < kI; ++i) {
        std::memset(c[i], 0, sizeof(float) * kJ);
    }

    int const vert_block_size = 1024;
    int const horz_block_size = 32;
    int vert_limit, horz_limit, vertical, horizontal, i, k, j;
    #pragma omp parallel for schedule(dynamic) private(vert_limit, horz_limit, vertical, horizontal, i, k, j)
    for(vertical = 0; vertical < kI; vertical += vert_block_size){
        for(horizontal = 0; horizontal < kK; horizontal += horz_block_size){
            vert_limit = vertical + vert_block_size <= kI ? (vertical + vert_block_size) : kI;
            //vert_limit = vertical + vert_block_size;
            for(i = vertical; i < vert_limit; i++){
                horz_limit = horizontal + horz_block_size <= kK ? (horizontal + horz_block_size) : kJ;
                //horz_limit = horizontal + horz_block_size;
                for(k = horizontal; k < horz_limit; k++){
                    for(j = 0; j < kJ; j++){
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
   }
}
