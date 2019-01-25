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
    int const vert_block_size = 8;
    int const horz_block_size = 8;
    #pragma omp parallel for
    for(int vertical = 0; vertical < kI; vertical += vert_block_size){
        for(int horizontal = 0; horizontal < kK; horizontal += horz_block_size){
            int vert_limit = std::min(vertical + vert_block_size, kI);
            // if(omp_get_thread_num() == 0){
            //     std::printf("vert lim %d \n", vert_limit);
            // }
            for(int i = vertical; i < vert_limit; i++){
                int horz_limit = std::min(horizontal + horz_block_size, kK);
                // if(omp_get_thread_num() == 0){
                //     std::printf("horz lim %d \n", horz_limit);
                // }
                for(int k = horizontal; k < horz_limit; k++){
                    for(int j = 0; j < kJ; j += 2){
                        c[i][j] += a[i][k] * b[k][j];
                        c[i][j+1] += a[i][k] * b[k][j+1];
                    }
                }
            }
        }
   }
}
