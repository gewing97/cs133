// Header inclusions, if any...

#include "gemm.h"
#include <vector>
#include <cstring>

// Using declarations, if any...

const int PROD_BLOCK_SIZE = 64;
const int MULT_BLOCK_SIZE = 20;
const int BLOCK_SIZE = 2;

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
    #pragma omp parallel for 
    for (int i = 0; i < kI; ++i) {
        std::memset(c[i], 0, sizeof(float) * kJ);
    }
    #pragma omp parallel for
    for(int prod_row = 0; prod_row < kI; prod_row += PROD_BLOCK_SIZE){
        for(int mult_row = 0; mult_row < kK; mult_row += MULT_BLOCK_SIZE){
            for(int j = 0; j < kJ; j++){
                for(int i = prod_row; i < prod_row + PROD_BLOCK_SIZE; i++){
                    for(int k = mult_row; k < mult_row + MULT_BLOCK_SIZE; k++){
                        c[i][j] = a[i][k] * b[k][j];
                    }
                }
            }
        }
    }
}
