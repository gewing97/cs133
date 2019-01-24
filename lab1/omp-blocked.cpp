// Header inclusions, if any...

#include "gemm.h"
#include <vector>
#include <cstring>

// Using declarations, if any...

const int PROD_BLOCK_SIZE = 512;
const int MULT_BLOCK_SIZE = 512;
const int BLOCK_SIZE = 2;





void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],  float c[kI][kJ]) {
    #pragma omp parallel for 
    for (int i = 0; i < kI; ++i) {
        std::memset(c[i], 0, sizeof(float) * kJ);
    }
    int const vert_block_size = 20;
    int const horz_block_size = 20;
    #pragma omp parallel for
    for(int vertical = 0; vertical < kI; vertical += vert_block_size){
        for(int horizontal = 0; horizontal < kJ; horizontal += horz_block_size){
            int vert_limit = std::min(vertical + vert_block_size, kI);
            for(int i = 0; i < vert_limit; i++){
                int horz_limit = std::min(horizontal + horz_block_size, kJ);
                for(int j = 0; j < horz_limit; j++){
                    for(int k = 0; k < vert_limit; k++){
                        c[i][j] = a[i][k] * b[k][j];
                    }
                }
            }
        }
    }
}








void GemmParallelBlocked_first_attempt(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
    #pragma omp parallel for 
    for (int i = 0; i < kI; ++i) {
        std::memset(c[i], 0, sizeof(float) * kJ);
    }
    #pragma omp parallel for
    for(int prod_row = 0; prod_row < kI; prod_row += PROD_BLOCK_SIZE){
        for(int mult_row = 0; mult_row < kK; mult_row += MULT_BLOCK_SIZE){
            for(int j = 0; j < kJ; j += 2){
                int end_prod_row = std::min(prod_row + PROD_BLOCK_SIZE, kI);
                for(int i = prod_row; i < end_prod_row; i += 2){
                    float temp_a = c[i][j];
                    float temp_b = c[i][j+1];
                    float temp_c = c[i+1][j];
                    float temp_d = c[i+1][j+1];
                    int end_mult_row = std::min(mult_row + MULT_BLOCK_SIZE, kK);
                    for(int k = mult_row; k < end_mult_row; k++){
                        temp_a += a[i][k] * b[k][j]; 
                        temp_b += a[i][k] * b[k][j+1];
                        temp_c += a[i+1][k] * b[k][j];
                        temp_d += a[i+1][k] * b[k][j+1];
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
