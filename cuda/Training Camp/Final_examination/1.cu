#include <stdio.h>
#include <stdlib.h>
#include "error.cuh"

#define N 3000 // Love u 3000 times!
#define BLOCK_SIZE 32

__managed__ int input_Matrix[N][N];
__managed__ int output_GPU[N][N];
__managed__ int output_CPU[N][N];
__global__ void gpu_ken(int input_M[N][N], int output_M[N][N])
{
    int x = blockIdx.x* blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x<N && y<N){
        if(input_M[y][x] > 100){
            output_M[y][x] = 0;
        }else{
            output_M[y][x] = input_M[y][x];
        }
    }

}
void cpu_ken(int input_M[N][N], int output_CPU[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if(input_M[i][j]>100)
            {
                output_CPU[i][j] = 0;
            }
            else
            {
                output_CPU[i][j] = input_Matrix[i][j];
            }
        }
    }
}

int main(int argc, char const* argv[])
{

    cudaEvent_t start, stop_gpu;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_gpu));


    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
        {

            input_Matrix[i][j] = rand() % 3001;
        }
    }
    cpu_ken(input_Matrix, output_CPU);

    CHECK(cudaEventRecord(start));
    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    printf("\n***********GPU RUN**************\n");
    gpu_ken<<<dimGrid, dimBlock >>>(input_Matrix, output_GPU);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));

    float elapsed_time_gpu;
    CHECK(cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu));
    printf("Time_GPU = %g ms.\n", elapsed_time_gpu);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop_gpu));

    int ok = 1;
    printf("\n***********Check result**************\n");
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (fabs(output_GPU[i][j] - output_CPU[i][j]) > (1.0e-10))
            {
                ok = 0;
                printf("cpu: %d; gpu: %d;\n", output_CPU[i][j], output_GPU[i][j]);
            }

        }
    }


    if (ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    // free memory
    return 0;
}