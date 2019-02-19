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

    float (*temp_c)[kJ] = new float[kI][kJ]();
    
    MPI_Request *a_requests;
    MPI_Request *b_requests;
    MPI_Request *c_requests;


    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int num_rows_per = kI/mpi_size;
    int offset = mpi_rank * num_rows_per;
    int vert_blocks_per = num_rows_per / VERT_BLOCK_SIZE;
    int horz_blocks_per = kK / HORZ_BLOCK_SIZE;

    if(mpi_rank == 0 && mpi_size > 1){
        b_requests = new MPI_Request[(mpi_size - 1) * horz_blocks_per];
        a_requests = new MPI_Request[(mpi_size - 1) * vert_blocks_per];
        c_requests = new MPI_Request[(mpi_size - 1) * vert_blocks_per];
        //MPI_Scatter(a, num_rows_per * kK, MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    	for(int i = 1; i < mpi_size; i++){
           for(int k = 0; k < horz_blocks_per; k++){
                MPI_Isend(b + (HORZ_BLOCK_SIZE * k), HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &b_requests[(i-1)*horz_blocks_per + k]);  
           }
	       for(int j = 0; j < vert_blocks_per; j++){
                MPI_Isend(a + (num_rows_per * i) + (VERT_BLOCK_SIZE * j), VERT_BLOCK_SIZE * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &a_requests[(i-1)*vert_blocks_per + j]);
                MPI_Irecv(c + (num_rows_per * i) + (VERT_BLOCK_SIZE * j), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &c_requests[(i-1)*vert_blocks_per + j]);
           }
        }
	    for (int i = 0; i < num_rows_per; ++i) {
            std::memset(c[offset + i], 0, sizeof(float) * kJ);
        }
	    // if(mpi_size > 1) MPI_Waitall(mpi_size - 1, b_requests, MPI_STATUSES_IGNORE);
    }
    else if(mpi_rank != 0){
	    float (*a_portion)[kK] = new float[kI][kK];
        float (*b_portion)[kJ] = new float[kK][kJ]; 
        a_requests = new MPI_Request[vert_blocks_per];
        c_requests = new MPI_Request[vert_blocks_per]; 
        b_requests = new MPI_Request[horz_blocks_per];
        //MPI_Scatter(NULL, num_rows_per * kK, MPI_FLOAT, a_portion + (num_rows_per * mpi_rank), num_rows_per * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
        for(int k = 0; k < horz_blocks_per; k++){
            MPI_Irecv(b_portion + (HORZ_BLOCK_SIZE * k), HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &b_requests[k]);
        }
        MPI_Recv(b_portion, kK*kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int j = 0; j < vert_blocks_per; j++){
            MPI_Irecv(a_portion + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * j), VERT_BLOCK_SIZE * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &a_requests[j]);
        }
        a = a_portion;
	    b = b_portion;
        c = new float[kI][kJ]();
    }

    int vert_limit, horz_limit, vertical, horizontal, i, k, j;
    int request_num_v = 0;
    int request_num_h = 0;
    for(vertical = offset; vertical < (offset + num_rows_per); vertical += VERT_BLOCK_SIZE){
        vert_limit = vertical + VERT_BLOCK_SIZE;
        if(mpi_rank != 0){
            printf("vertical %12X\n", &a_requests[request_num_v]);
            MPI_Wait(&a_requests[request_num_v], MPI_STATUS_IGNORE);
            if(request_num_v > 0){
                MPI_Isend(c + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * (request_num_v - 1)), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &c_requests[request_num_v - 1]);
            }
            request_num_v++;
        }
        for(horizontal = 0; horizontal < kK; horizontal += HORZ_BLOCK_SIZE){
            horz_limit = horizontal + HORZ_BLOCK_SIZE;
            if(mpi_rank != 0 && request_num_h < horz_blocks_per){
                printf("horizontal %12X\n", &b_requests[request_num_h]);
                MPI_Wait(&b_requests[request_num_h], MPI_STATUSES_IGNORE);
                request_num_h++;
            }
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
    printf("1\n");

    if(mpi_rank == 0 && mpi_size > 1){
        MPI_Waitall((mpi_size - 1) * vert_blocks_per, a_requests, MPI_STATUSES_IGNORE);
        printf("2\n");
        MPI_Waitall((mpi_size - 1) * horz_blocks_per, b_requests, MPI_STATUSES_IGNORE);
        printf("3\n");
        MPI_Waitall((mpi_size - 1) * vert_blocks_per, c_requests, MPI_STATUSES_IGNORE);
        printf("4\n");
    }
    else if(mpi_rank != 0){
        MPI_Isend(c + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * (vert_blocks_per - 1)), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &c_requests[vert_blocks_per - 1]);
        MPI_Waitall(vert_blocks_per, c_requests, MPI_STATUSES_IGNORE);
    }  
}
