#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int N = 11; //size of a
int m = 5; //size of h

void histogram(int *a, int *h){
    for(int i = 0; i < m; i++){
        h[i] = 0; //initialize the array
    }
    for(int i = 0; i < N; i++){ //N is the number of elements that THIS PARTICULAR PROCESS HAS
        h[a[i] - 1]++; //compute the proper values within each process don't know what m is though
    }
    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0){  //to prevent unnecessary computation, only the root processor will compute final histogram
        int * rec_buffer = (int *) malloc(num_proc * m * sizeof(int));
        MPI_Gather(h, m, MPI_INT, rec_buffer, m, MPI_INT, 0, MPI_COMM_WORLD);
        int * result = (int *) calloc(m, sizeof(int));
        for(int i = 0; i < m; i++){
            h[i] = 0;
            for(int j = 0; j < num_proc; j++){
                h[i] += rec_buffer[j*m + i];
            }
        }
    } else{
        MPI_Gather(h, m, MPI_INT, NULL, m, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

int main(){
    MPI_Init(NULL, NULL);
    int a[] = {1,2,3,4,5,3,2,1,2,3,4};
    int h[5];
    histogram(a, h);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0){
        for(int i = 0; i < m; i++){
            printf("%d\n", h[i]);
        }
    }
    MPI_Finalize();
}