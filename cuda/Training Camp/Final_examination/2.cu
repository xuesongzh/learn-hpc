#include <stdio.h>
#include <math.h>

#define N 100000000
#define BLOCK_SIZE 256
#define GRID_SIZE 32
#define topk 10

__managed__ int source[N];
__managed__ int gpu_result[topk];
__managed__ int _1_pass_result[topk * GRID_SIZE];

// Insert a data into an array (containing k values), still keeping the order of the array from max to min
__device__ __host__ void insert_value(int *array, int k, int data)
{
    for(int i=0; i<k; i++)
    {
        if(array[i] == data)
        {
            return;
        }
    }
    if(data < array[k-1])
    {
        return;
    }
    //19, 18, 17, 16,.........4, 3, 2, 1, 0
    for(int i = k-2; i>=0; i--)
    {
        if(data > array[i])
        {
            array[i + 1] = array[i];
        }
        else
        {
            array[i+1] = data;
            return;
        }
    }
    
    array[0] = data;
}

__global__ void gpu_topk(int *input, int *output, int length, int k)
{
    // my code 
    int array[topk]; 
    for (int i = 0; i < topk; i++)
    {
        array[i] = INT_MIN;
    }
    
    __shared__ int sh [BLOCK_SIZE * topk];
    
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length; idx += gridDim.x * blockDim.x)
    {
        insert_value(array, k, input[idx]);
    }
    for (int j = 0; j < topk; j++)
    {
        sh[topk * threadIdx.x + j] = array[j];
    }
    __syncthreads();

    for(int i = BLOCK_SIZE / 2; i >= 1; i /= 2)
    {
        if (threadIdx.x < i)
        {
            for (int m = 0; m < topk; m++)
            {
                insert_value(array, topk, sh[topk * (threadIdx.x + i) + m]);
            }
        }            
        __syncthreads();
        if(threadIdx.x < i)
        {
            for (int m = 0; m < topk; m++)
            {
                sh[topk* threadIdx.x + m] = array[m];
            }
        }
        __syncthreads();
    }
    
    if (blockIdx.x * blockDim.x < length)
    {
        if (threadIdx.x == 0)
        {
            for (int m = 0; m < topk; m++)
            {
                output[topk * blockIdx.x + m] = sh[m];
            }
        }
    }

}

void cpu_topk(int *input, int *output, int length, int k)
{
    for(int i =0; i< length; i++)
    {
        insert_value(output, k, input[i]);
    }
}

int main()
{
    printf("Init source data...........\n");
    for(int i=0; i<N; i++)
    {
        source[i] = rand();
    }

    printf("Complete init source data.....\n");
    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);
    printf("GPU Run **************\n");
    for(int i =0; i<20; i++)
    {
        gpu_topk<<<GRID_SIZE, BLOCK_SIZE>>>(source, _1_pass_result, N, topk);

        gpu_topk<<<1, BLOCK_SIZE>>>(_1_pass_result, gpu_result, topk * GRID_SIZE, topk);
        // gpu_topk<<<1, BLOCK_SIZE>>>(source, gpu_result, N, topk);

        cudaDeviceSynchronize();
    }
    printf("GPU Complete!!!\n");
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    int cpu_result[topk] ={0};
    printf("CPU RUN***************\n");
    cpu_topk(source, cpu_result, N, topk);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    printf("CPU Complete!!!!!");

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

    bool error = false;
    for(int i =0; i<topk; i++)
    {
        printf("CPU top%d: %d; GPU top%d: %d;\n", i+1, cpu_result[i], i+1, gpu_result[i]);
        if(fabs(gpu_result[i] - cpu_result[i]) > 0)
        {
            error = true;
        }
    }
    printf("Result: %s\n", (error?"Error":"Pass"));
    printf("CPU time: %.2f; GPU time: %.2f\n", time_cpu, (time_gpu/20.0));
}




