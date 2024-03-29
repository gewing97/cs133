// Header inclusions, if any...

#include "gemm.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <omp.h>

// Using declarations, if any...

#define VERT_BLOCK_SIZE 64
#define HORZ_BLOCK_SIZE 32
#define NUM_THREADS 8

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],  float c[kI][kJ]) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < kI; ++i) {
        std::memset(c[i], 0, sizeof(float) * kJ);
    }

    int vert_limit, horz_limit, vertical, horizontal, i, k, j;
    #pragma omp parallel for schedule(static) private(vert_limit, horz_limit, vertical, horizontal, i, k, j) num_threads(NUM_THREADS)
    for(vertical = 0; vertical < kI; vertical += VERT_BLOCK_SIZE){
        //vert_limit = vertical + VERT_BLOCK_SIZE <= kI ? (vertical + VERT_BLOCK_SIZE) : kI;
        vert_limit = vertical + VERT_BLOCK_SIZE;
        for(horizontal = 0; horizontal < kK; horizontal += HORZ_BLOCK_SIZE){
            //horz_limit = horizontal + HORZ_BLOCK_SIZE <= kK ? (horizontal + HORZ_BLOCK_SIZE) : kJ;
            horz_limit = horizontal + HORZ_BLOCK_SIZE;
            for(i = vertical; i < vert_limit; i++){
                for(k = horizontal; k < horz_limit; k+=8){
                    for(j = 0; j < kJ; j++){
                        c[i][j] += 
                            (a[i][k] * b[k][j]) + (a[i][k+1] * b[k+1][j]) + 
                            (a[i][k+2] * b[k+2][j]) + (a[i][k+3] * b[k+3][j]) + 
                            (a[i][k+4] * b[k+4][j]) + (a[i][k+5] * b[k+5][j]) + 
                            (a[i][k+6] * b[k+6][j]) + (a[i][k+7] * b[k+7][j]);
                    }
                }
            }
        }
   }
}
