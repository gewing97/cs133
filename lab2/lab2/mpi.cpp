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

    // float (*temp_c)[kJ] = new float[kI][kJ]();
    
    MPI_Request *a_requests;
    MPI_Request *b_requests;
    MPI_Request *c_requests;

    float (*a_portion)[kK] = new float[kI][kK];
    float (*b_portion)[kJ] = new float[kK][kJ]; 


    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int num_rows_per = kI/mpi_size;
    int offset = mpi_rank * num_rows_per;
    int vert_blocks_per = num_rows_per / VERT_BLOCK_SIZE;
    int horz_blocks_per = kK / HORZ_BLOCK_SIZE;

    if(mpi_rank == 0 && mpi_size > 1){
        b_requests = new MPI_Request[(mpi_size - 1)];
        a_requests = new MPI_Request[(mpi_size - 1)];
        c_requests = new MPI_Request[(mpi_size - 1)];
    	for(int i = 1; i < mpi_size; i++){
        //    for(int k = 0; k < horz_blocks_per; k++){
        //         MPI_Isend(b + (HORZ_BLOCK_SIZE * k), HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &b_requests[(i-1)*horz_blocks_per + k]);  
        //    }
            MPI_Isend(b + (HORZ_BLOCK_SIZE * 0), HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &b_requests[i-1]);
	       // for(int j = 0; j < vert_blocks_per; j++){
        //         MPI_Isend(a + (num_rows_per * i) + (VERT_BLOCK_SIZE * j), VERT_BLOCK_SIZE * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &a_requests[(i-1)*vert_blocks_per + j]);
        //         MPI_Irecv(c + (num_rows_per * i) + (VERT_BLOCK_SIZE * j), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &c_requests[(i-1)*vert_blocks_per + j]);
        //    }
            MPI_Isend(a + (num_rows_per * i) + (VERT_BLOCK_SIZE * 0), VERT_BLOCK_SIZE * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &a_requests[i-1]);
            MPI_Irecv(c + (num_rows_per * i) + (VERT_BLOCK_SIZE * 0), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &c_requests[i-1]);
        }
	    for (int i = 0; i < num_rows_per; ++i) {
            std::memset(c[offset + i], 0, sizeof(float) * kJ);
        }
    }
    else if(mpi_rank != 0){
        a_requests = new MPI_Request;
        c_requests = new MPI_Request; 
        b_requests = new MPI_Request;
        // for(int k = 0; k < horz_blocks_per; k++){
        //     MPI_Irecv(b_portion + (HORZ_BLOCK_SIZE * k), HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &b_requests[k]);
        // }
        MPI_Irecv(b_portion + (HORZ_BLOCK_SIZE * 0), HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, b_requests);
        // for(int j = 0; j < vert_blocks_per; j++){
        //     MPI_Irecv(a_portion + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * j), VERT_BLOCK_SIZE * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &a_requests[j]);
        // }
        MPI_Irecv(a_portion + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * 0), VERT_BLOCK_SIZE * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, a_requests);
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
            MPI_Wait(a_requests, MPI_STATUS_IGNORE);
            a_requests = new MPI_Request;
            if(request_num_v > 0){
                if(request_num_v > 1){
                    MPI_Wait(c_requests, MPI_STATUS_IGNORE);
                    c_requests = new MPI_Request;
                }
                MPI_Isend(c + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * (request_num_v - 1)), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, c_requests);
            }
            if(request_num_v + 1 < vert_blocks_per){
                MPI_Irecv(a_portion + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * (request_num_v + 1)), VERT_BLOCK_SIZE * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, a_requests);
            }
        } else if(mpi_size > 1){
            for(int proc = 1; proc < mpi_size; proc++){
                MPI_Wait(&a_requests[proc-1], MPI_STATUS_IGNORE);
                a_requests[proc-1] = new MPI_Request;
                if(request_num_v > 0){
                    // if(request_num_v > 1){
                        MPI_Wait(&c_requests[proc-1], MPI_STATUS_IGNORE);
                        c_requests[proc-1] = new MPI_Request; 
                    }
                MPI_Irecv(c + (num_rows_per * proc) + (VERT_BLOCK_SIZE * (request_num_v)), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &c_requests[proc-1]);
                // }
                if(request_num_v + 1 < vert_blocks_per){
                    MPI_Isend(a + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * request_num_v + 1), VERT_BLOCK_SIZE * kK, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &a_requests[(proc-1)*vert_blocks_per + request_num_h + 1]);
                }    
            }
        }
        request_num_v++;
        for(horizontal = 0; horizontal < kK; horizontal += HORZ_BLOCK_SIZE){
            horz_limit = horizontal + HORZ_BLOCK_SIZE;
            if(request_num_h < horz_blocks_per){
                if(mpi_rank != 0){
                    //printf("horizontal %d %d %12X\n", mpi_rank, request_num_h, &b_requests[request_num_h]);
                    MPI_Wait(b_requests, MPI_STATUS_IGNORE);
                    b_requests = new MPI_Request;
                    if(request_num_h + 1 < horz_blocks_per){
                        MPI_Irecv(b_portion + (HORZ_BLOCK_SIZE * request_num_h), HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, b_requests);
                    }
                } else if (mpi_size > 1){
                    for(int proc = 1; proc < mpi_size; proc++){
                        MPI_Wait(&b_requests[proc-1], MPI_STATUS_IGNORE);
                        b_requests[proc-1] = new MPI_Request;
                        if(request_num_h + 1 < horz_blocks_per){
                            MPI_Isend(b + (HORZ_BLOCK_SIZE * request_num_h), HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &b_requests[proc-1]);  
                        }
                    }
                }
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

    if(mpi_rank == 0 && mpi_size > 1){
        // MPI_Waitall((mpi_size - 1) * horz_blocks_per, b_requests, MPI_STATUSES_IGNORE);
        // MPI_Waitall((mpi_size - 1) * vert_blocks_per, a_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(mpi_size - 1, c_requests, MPI_STATUSES_IGNORE);
    }
    else if(mpi_rank != 0){
        MPI_Send(c + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * (vert_blocks_per - 1)), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        printf("send me daddy %d\n", mpi_rank);
        // MPI_Waitall(vert_blocks_per, c_requests, MPI_STATUSES_IGNORE);
        //printf("waiting on the world to change %d\n", mpi_rank);
    }  
}
