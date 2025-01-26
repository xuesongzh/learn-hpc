#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 16

__global__ void gpu_transpose(int *in,int *out, int width)
{ 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( y < width && x < width)
    {
        out[x * width + y] = in[y * width + x];
    }
} 

void cpu_matrix_transpose(int *in, int *out, int width)
{
    for(int y = 0; y < width; y++)
    {
        for(int x = 0; x < width; x++)
        {
            out[x * width + y] = in[y * width + x];
        }
    }
}

int main(int argc, char const *argv[])
{
    int m=1000;
    int *h_in = (int*)malloc(sizeof(int)*m*m);
    int *h_out = (int*)malloc( sizeof(int)*m*m );
    int *h_cpu_out = (int*)malloc( sizeof(int)*m*m);
    

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            h_in[i * m + j] = rand() % 1024;
        }
    }


    int *d_out, *d_in;
    cudaMalloc((void **) &d_out, sizeof(int)*m*m);
    cudaMalloc((void **) &d_in, sizeof(int)*m*m);


    // copy matrix A and B from host to device memory
    cudaMemcpy(d_in, h_in, sizeof(int)*m*m, cudaMemcpyHostToDevice);


    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    gpu_transpose<<<dimGrid, dimBlock>>>(d_in, d_out, m);    

    cudaMemcpy(h_out, d_out, sizeof(int)*m*m, cudaMemcpyDeviceToHost);
    //cudaThreadSynchronize();

    cpu_matrix_transpose(h_in, h_cpu_out,m);

    int ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if(fabs(h_out[i*m + j] - h_cpu_out[i*m + j])>(1.0e-10))
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

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}