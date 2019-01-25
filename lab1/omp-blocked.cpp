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
    #pragma omp parallel for schedule(dynamic)
    for(int vertical = 0; vertical < kI; vertical += vert_block_size){
        for(int horizontal = 0; horizontal < kK; horizontal += horz_block_size){
            //int vert_limit = vertical + vert_block_size <= kI ? (vertical + vert_block_size) : kI;
            int vert_limit = vertical + vert_block_size;
	    for(int i = vertical; i < vert_limit; i++){
                //int horz_limit = horizontal + horz_block_size <= kK ? (horizontal + horz_block_size) : kJ;
                int horz_limit = horizontal + horz_block_size;
		for(int k = horizontal; k < horz_limit; k++){
                    for(int j = 0; j < kJ; j++){
                        c[i][j] += a[i][k] * b[k][j];
		    }
                }
            }
        }
   }
}
