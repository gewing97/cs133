// Header inclusions, if any...

#include "gemm.h"
#include <vector>
#include <cstring>

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {

    #pragma omp parallel for 
    for (int i = 0; i < kI; ++i) {
        std::memset(c[i], 0, sizeof(float) * kJ);
    }
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < kI; ++i) {
        for (int k = 0; k < kK; ++k) {
            for (int j = 0; j < kJ; ++j) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
