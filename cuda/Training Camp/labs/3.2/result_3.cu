#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 16

#define M 100
#define N 100
#define K 100
__managed__ int a[M * N];
__managed__ int b[N * K];
__managed__ int c[M * K];

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{

    int *h_cc = (int*)malloc(sizeof(int)*M*K);
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));


    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i * N + j] = rand() % 1024;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            b[i * K + j] = rand() % 1024;
        }
    }

    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    CHECK(cudaEventRecord(start));
    //cudaEventQuery(start);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(a, b, c, M, N, K);    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms.\n", elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    //cudaThreadSynchronize();
    

    cpu_matrix_mult(a, b, h_cc, M, N, K);

    int ok = 1;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            if(fabs(h_cc[i*K + j] - c[i*K + j])>(1.0e-10))
            {
                
                ok = 0;
            }
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    return 0;
}