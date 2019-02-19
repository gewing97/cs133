// Header inclusions, if any...

#include <mpi.h>
#include <vector>
#include <cstring>
#include <iostream>
#include "../lab1/gemm.h"

// Using declarations, if any...
#define VERT_BLOCK_SIZE 64
#define HORZ_BLOCK_SIZE 32


void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
    int mpi_size;
    int mpi_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int num_rows_per = kI/mpi_size;

    if(mpi_rank == 0){
        for(int i = 1; i < mpi_size; i++){
            MPI_Send(a + (num_rows_per * i), num_rows_per * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            MPI_Send(b, kK*kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Recv(a + (num_rows_per * mpi_rank), num_rows_per * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, kK*kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int offset = num_rows_per * mpi_rank;
    for (int i = 0; i < num_rows_per; ++i) {
        std::memset(c[offset + i], 0, sizeof(float) * kJ);
    }


    int vert_limit, horz_limit, vertical, horizontal, i, k, j;
    for(vertical = offset; vertical < num_rows_per; vertical += VERT_BLOCK_SIZE){
        vert_limit = vertical + VERT_BLOCK_SIZE;
        for(horizontal = 0; horizontal < kK; horizontal += HORZ_BLOCK_SIZE){
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
    if(mpi_rank == 0){
        for(int i = 1; i < mpi_size; i++){
            MPI_Recv(c + (num_rows_per * i), num_rows_per * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    }
    else{
        MPI_Send(c + (num_rows_per * mpi_rank), num_rows_per * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }  

}