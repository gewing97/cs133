// Header inclusions, if any...

#include "gemm.h"
#include <vector>
#include <cstring>

// Using declarations, if any...

const int PROD_BLOCK_SIZE = 16;
const int MULT_BLOCK_SIZE = 16;
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
            for(int j = 0; j < kJ; j += 2){
                for(int i = prod_row; i < prod_row + PROD_BLOCK_SIZE; i += 2){
                    float temp_a = c[i][j];
                    float temp_b = c[i][j+1];
                    float temp_c = c[i+1][j];
                    float temp_d = c[i+1][j+1];
                    for(int k = mult_row; k < mult_row + MULT_BLOCK_SIZE; k++){
                        temp_a += a[i][k] * b[k][j]; 
                        temp_b += a[i][k] * b[k][j+1];
                        temp_c = a[i+1][k] * b[k][j];
                        temp_d = c[i+1][k] * b[k][j+1];
                    }
                    c[i][j] = temp_a;
                    c[i][j+1] = temp_b;
                    c[i+1][j] = temp_c;
                    c[i+1][j+1] = temp_d;
                }
            }
        }
    }
}
