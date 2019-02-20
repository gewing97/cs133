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
            MPI_Isend(b, HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &b_requests[i-1]);
            MPI_Isend(a + (num_rows_per * i), VERT_BLOCK_SIZE * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &a_requests[i-1]);
        }
	    for (int i = 0; i < num_rows_per; ++i) {
            std::memset(c[offset + i], 0, sizeof(float) * kJ);
        }
    }
    else if(mpi_rank != 0){
        a_requests = new MPI_Request;
        c_requests = new MPI_Request; 
        b_requests = new MPI_Request;
        MPI_Irecv(b_portion, HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, b_requests);
        MPI_Irecv(a_portion + (num_rows_per * mpi_rank), VERT_BLOCK_SIZE * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, a_requests);
        a = a_portion;
	    b = b_portion;
        c = new float[kI][kJ]();
    }

    int vert_limit, horz_limit, vertical, horizontal, i, k, j;
    // int request_num_v = 0;
    // int request_num_h = 0;
    for(vertical = offset; vertical < (offset + num_rows_per); vertical += VERT_BLOCK_SIZE){
        vert_limit = vertical + VERT_BLOCK_SIZE;
        if(mpi_rank != 0){
            //get new portion of a 
            MPI_Wait(a_requests, MPI_STATUS_IGNORE);
            //send processed portion of c
            if(vertical > offset){
                if(vertical > offset + VERT_BLOCK_SIZE){
                    MPI_Wait(c_requests, MPI_STATUS_IGNORE);
                    c_requests = new MPI_Request;
                }
                MPI_Isend(c + vertical - VERT_BLOCK_SIZE, VERT_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, c_requests);
            }
            //request next portion of a
            if(vert_limit + VERT_BLOCK_SIZE < (offset + num_rows_per)){
                a_requests = new MPI_Request;
                MPI_Irecv(a_portion + vert_limit, VERT_BLOCK_SIZE * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, a_requests);
            }
        } else if(mpi_size > 1){
            printf("a_requests: %12X c_requests: %12X\n", a_requests, c_requests);
            //send current portion of a
            MPI_Waitall(mpi_size-1, a_requests, MPI_STATUSES_IGNORE);
            a_requests = new MPI_Request[mpi_size-1];
            //receive c
            if(vertical > offset + VERT_BLOCK_SIZE){
                MPI_Waitall(mpi_size-1, c_requests, MPI_STATUSES_IGNORE);
                c_requests = new MPI_Request[mpi_size-1];
            }
            for(int proc = 1; proc < mpi_size; proc++){
                //receive last part of c
                if (vertical > offset){
                    MPI_Irecv(c + vertical - VERT_BLOCK_SIZE, VERT_BLOCK_SIZE * kJ, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &c_requests[proc-1]);
                }
                //send next portion of a
                if(vert_limit + VERT_BLOCK_SIZE < (offset + num_rows_per)){
                    MPI_Isend(a + ((proc -1) * vert_blocks_per) + (vert_limit - offset), VERT_BLOCK_SIZE * kK, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &a_requests[proc-1]);
                }    
            }
        }
        for(horizontal = 0; horizontal < kK; horizontal += HORZ_BLOCK_SIZE){
            horz_limit = horizontal + HORZ_BLOCK_SIZE;
            if(vertical == offset){
                if(mpi_rank != 0){
                    MPI_Wait(b_requests, MPI_STATUS_IGNORE);
                    if(horz_limit + HORZ_BLOCK_SIZE < kK){
                        b_requests = new MPI_Request;
                        MPI_Irecv(b_portion + horz_limit, HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, b_requests);
                    }
                } else if (mpi_size > 1){
                    printf("b_requests: %12X\n", b_requests);
                    MPI_Waitall(mpi_size - 1, b_requests, MPI_STATUSES_IGNORE);
                    if(horz_limit + HORZ_BLOCK_SIZE < kK){
                        b_requests = new MPI_Request[mpi_size-1];
                        for(int proc = 1; proc < mpi_size; proc++){
                            MPI_Isend(b + horz_limit, HORZ_BLOCK_SIZE * kJ, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &b_requests[proc-1]);  
                        }
                    }
                }
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
        for(int proc = 1; proc < mpi_size; proc++){
            MPI_Recv(c + (num_rows_per * proc) + (VERT_BLOCK_SIZE * (vert_blocks_per - 1)), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else if(mpi_rank != 0){
        MPI_Send(c + (num_rows_per * mpi_rank) + (VERT_BLOCK_SIZE * (vert_blocks_per - 1)), VERT_BLOCK_SIZE * kJ, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        printf("send me daddy %d\n", mpi_rank);
    }  
}
