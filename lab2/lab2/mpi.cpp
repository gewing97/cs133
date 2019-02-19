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
    printf("fucking christ\n");
    
    int mpi_size;
    int mpi_rank;

    float (*temp_c)[kJ] = new float[kI][kJ]();
    
    MPI_Request *requests;
    
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int num_rows_per = kI/mpi_size;
    int offset = mpi_rank * num_rows_per;
    if(mpi_rank == 0 && mpi_size > 1){
        MPI_Request *requests = new MPI_Request[mpi_size - 1];
        MPI_Scatter(a, num_rows_per * kK, MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    	for(int i = 1; i < mpi_size; i++){
	       printf("sending to %d\n", i);
	       MPI_Isend(b, kK*kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[i-1]);	
	    }
	    for (int i = 0; i < num_rows_per; ++i) {
            std::memset(c[offset + i], 0, sizeof(float) * kJ);
        }
	    if(mpi_size > 1) MPI_Waitall(mpi_size - 1, requests, MPI_STATUSES_IGNORE);
    }
    else if(mpi_rank != 0){
	    float (*a_portion)[kK] = new float[kI][kK];
        float (*b_portion)[kJ] = new float[kK][kJ]; 
	    MPI_Scatter(NULL, num_rows_per * kK, MPI_FLOAT, a_portion + (num_rows_per * mpi_rank), num_rows_per * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Recv(b_portion, kK*kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("process %d received\n",mpi_rank);
        a = a_portion;
	    b = b_portion;
        c = new float[kI][kJ]();
    }

    printf("%d finished setup\n", mpi_rank);

    int vert_limit, horz_limit, vertical, horizontal, i, k, j;
    for(vertical = offset; vertical < (offset + num_rows_per); vertical += VERT_BLOCK_SIZE){
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

    if(mpi_rank == 0 && mpi_size > 1){
        MPI_Request *requests = new MPI_Request[mpi_size - 1];
        for(int i = 1; i < mpi_size; i++){
            MPI_Irecv(c + (num_rows_per * i), num_rows_per * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[i-1]);
	    }
    	MPI_Waitall(mpi_size-1, requests, MPI_STATUSES_IGNORE);
    }
    else if(mpi_rank != 0){
        MPI_Send(c + (num_rows_per * mpi_rank), num_rows_per * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }  
}
