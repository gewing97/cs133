#include <stdio.h>
#include <omp.h>
int N = 11;
int m = 5;

void histogram(int *a, int *h){
    omp_lock_t histo_locks[m];
    #pragma omp parallel for 
    for(int i = 0; i < m; i++){
        h[i] = 0;
        omp_init_lock(&histo_locks[i]);
    }
    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        omp_set_lock(&histo_locks[a[i]]);
        h[a[i] -1]++;
        omp_unset_lock(&histo_locks[a[i]]);
    }
}

int main(){
    int a[] = {1,2,3,4,5,3,2,1,2,3,4};
    int h[5];
    histogram(a, h);
    for(int i = 0; i <m; i++){
        printf("%d\n", h[i]);
    }

}
