#include <stdio.h>
#include <math.h>

// a[][] * b[][] = c[][]
// 
//                         b00 b01 b02 b03
//                         b10 b11 b12 b13
//                         b20 b21 b22 b23
//                         b30 b31 b32 b33
//
// a00 a01 a02 a03         c00 c01 c02 c03
// a10 a11 a12 a13         c10 c11 c12 c13     block(1, 0) -> shared memory
// a20 a21 a22 a23         c20 c21 c22 c23     c20 c21
// a30 a31 a32 a33         c30 c31 c32 c33     c30 c31
//
//                              b00 b01->  sub_b_step_0
//                              b10 b11
//
//                              b20 b21->  sub_b_step_1
//                              b30 b31
// sub_a_step_0 sub_a_step_1    sub_c
// a20 a21      a22 a23         c20 c21
// a30 a31      a32 a33         c30 c31
//
// sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;
//
// for(int step =0; step < N/block_size; step++ )
//      load sub_a_step to shared memory;
//      load sub_b_step to shared memory;
//      tmp += sub_a_step_on_sharedmemory * sub_b_step_on_sharedmemory;
// sub_c = tmp;
//
// cudaMalloc -> global memory
// data global memory -> shared memory
// threads shared memory -> register
// shared memory SM(stream multi-processor) same block same shared memory
//
// c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
// a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
// 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
// b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33

#define M 1000
#define N 500
#define K 1000

__managed__ int a[M*N];
__managed__ int b[N*K];
__managed__ int c_gpu[M*K];
__managed__ int c_cpu[M*K];

#define BLOCK_SIZE 16

__global__ void gpu_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tmp =0;
    int idx;
    for(int step=0; step <= n/BLOCK_SIZE; step++)
    {
        int step_x = step * BLOCK_SIZE + threadIdx.x;
        int step_y = y;
        idx = step_y * n + step_x;
        if(step_x >= n || step_y >= m)
        {
            sub_a[threadIdx.y][threadIdx.x] =0;
        }
        else
        {
            sub_a[threadIdx.y][threadIdx.x] = a[idx];
        }

        step_x = x;
        step_y = step * BLOCK_SIZE + threadIdx.y;
        idx = step_y * k +step_x;
        if(step_x >= k || step_y >= n)
        {
            sub_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            sub_b[threadIdx.y][threadIdx.x] = b[idx];
        }

        __syncthreads();

        for(int i = 0; i < BLOCK_SIZE; i++)
        {
            tmp +=sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
        }
        __syncthreads();
    }

    if ( x < k && y < m)
    {
        c[y*k + x] = tmp; 
    }
}

void cpu_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    for( int y = 0; y < m; y++)
    {
        for(int x = 0; x < k; x++)
        {
            int tmp = 0;
            for(int step =0; step < n; step++)
            {
                tmp += a[y*n + step] * b[step*k + x];
            }
            c[y * k + x] = tmp;
        }
    }
}

int main()
{

    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);


    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            a[y * N + x] = rand()%1024;
        }
    }

    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<K; ++x)
        {
            b[y*K + x] = rand()%1024;
        }
    }

    unsigned int grid_x = (K + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_y = (M + BLOCK_SIZE -1)/BLOCK_SIZE;

    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    cudaEventRecord(start);

    gpu_matrix<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cpu_matrix(a, b, c_cpu, M, N, K);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    bool errors = false;
    float time_cpu, time_gpu;

    for(int y=0; y<M; y++)
    {
        for(int x=0; x<K; x++)
        {
            if(fabs(c_cpu[y*K + x] - c_gpu[y*K+x]) > (1.0e-10))
            {
                errors = true;
                printf("c_cpu: %d. c_gpu: %d", c_cpu[y*K + x], c_gpu[y*K+x]);
            }
        }
    }

    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
    cudaEventElapsedTime(&time_gpu, start ,stop_gpu);

    printf("Result: %s\n", errors?"Error":"Pass");
    printf("CPU time: %.2f; GPU time: %.2f;\n", time_cpu, time_gpu);

    return 0;
}


























