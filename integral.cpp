#include <math.h>
#include <stdio.h>

double integral(int number_steps){
        const double increment = 1.0 / number_steps;
        double sum = 0;
        #pragma omp parallel for num_threads(16)
        for(int i = 1; i < number_steps; i++){
                double x = increment * i; 
                double y = increment * (i-1);
                double midpoint = (x + y) / 2;
                sum += (sqrt(midpoint)/(1 + midpoint*midpoint*midpoint)) * (x - y);
        }
        return sum;
}

int main(){
        printf("%f\n", integral(1048576));
}

#include <math.h>
#include <stdio.h>

double integral(int number_steps){
        const double increment = 1.0 / number_steps;
        double sum = 0;
        #pragma omp parallel for num_threads(16)
        for(int i = 1; i < number_steps; i++){
                double x = increment * i; 
                double y = increment * (i-1);
                double midpoint = (x + y) / 2;
                #pragma omp critical
                sum += (sqrt(midpoint)/(1 + midpoint*midpoint*midpoint)) * (x - y);
        }
        return sum;
}

int main(){
        printf("%f\n", integral(1048576));
}

#include <math.h>
#include <stdio.h>

double integral(int number_steps){
        const double increment = 1.0 / number_steps;
        double sum = 0;
        #pragma omp parallel for reduction(+:sum) num_threads(16)
        for(int i = 1; i < number_steps; i++){
                double x = increment * i; 
                double y = increment * (i-1);
                double midpoint = (x + y) / 2;
                sum += (sqrt(midpoint)/(1 + midpoint*midpoint*midpoint)) * (x - y);
        }
        return sum;
}


int main(){
        printf("%f\n", integral(1048576));
}