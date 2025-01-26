/* Code to perform Hadamard (elementwise) multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

// Define the size of the arrays to be computed
#define NROWS_F 520
#define NCOLS_F 1032

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

// Bring in helper header to work with matrices
#include "mat_helper.hpp"

int main(int argc, char** argv) {

    // Parse arguments and set the target device
    cl_device_type target_device;
    cl_uint dev_index = h_parse_args(argc, argv, &target_device);
    
    // Useful for checking OpenCL errors
    cl_int errcode;

    // Create handles to platforms, 
    // devices, and contexts

    // Number of platforms discovered
    cl_uint num_platforms;

    // Number of devices discovered
    cl_uint num_devices;

    // Pointer to an array of platforms
    cl_platform_id *platforms = NULL;

    // Pointer to an array of devices
    cl_device_id *devices = NULL;

    // Pointer to an array of contexts
    cl_context *contexts = NULL;
    
    // Helper function to acquire devices
    h_acquire_devices(target_device,
                     &platforms,
                     &num_platforms,
                     &devices,
                     &num_devices,
                     &contexts);
    
    // Number of command queues to generate
    cl_uint num_command_queues = num_devices;
    
    // Do we enable out-of-order execution 
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling?
    cl_bool profiling = CL_TRUE;
    
    // Create the command queues
    cl_command_queue* command_queues = h_create_command_queues(
        devices,
        contexts,
        num_devices,
        num_command_queues,
        ordering,
        profiling
    );

    // Choose the first available context 
    // and compute device to use
    assert(dev_index < num_devices);
    cl_context context = contexts[dev_index];
    cl_command_queue command_queue = command_queues[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // D, E, F is of size (N0_F, N1_F)
    cl_uint N0_F = NROWS_F, N1_F = NCOLS_F;

    // Number of bytes in each matrix
    size_t nbytes_D=N0_F*N1_F*sizeof(float);   
    size_t nbytes_E=N0_F*N1_F*sizeof(float);
    size_t nbytes_F=N0_F*N1_F*sizeof(float);

    // Allocate memory for matrices A, B, and C on the host
    cl_float* D_h = (cl_float*)h_alloc(nbytes_D);
    cl_float* E_h = (cl_float*)h_alloc(nbytes_E);
    cl_float* F_h = (cl_float*)h_alloc(nbytes_F);

    // Fill host matrices with random numbers in the range 0, 1
    m_random(D_h, N0_F, N1_F);
    m_random(E_h, N0_F, N1_F);

    // Make Buffers on the compute device for matrices D, E, and F
    cl_mem D_d = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_D, 
                                     NULL, 
                                     &errcode);
    H_ERRCHK(errcode);
    
    cl_mem E_d = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_E, 
                                     NULL, 
                                     &errcode);
    H_ERRCHK(errcode);
    
    cl_mem F_d = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_F, 
                                     NULL, 
                                     &errcode);
    H_ERRCHK(errcode);

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_elementwise.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);

    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_elementwise", &errcode);
    H_ERRCHK(errcode);
    
    // Set arguments to the kernel (not thread safe)
    H_ERRCHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &D_d));
    H_ERRCHK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &E_d));
    H_ERRCHK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &F_d));
    H_ERRCHK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &N0_F));
    H_ERRCHK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &N1_F));

    // Write memory from the host
    // to D_d and E_d on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
   
    // Event to store profiling information
    cl_event io_event;

    //// Use clEnqueueWriteBuffer to copy memory ////
    //// from the host to the buffer ////
    
    H_ERRCHK(
        clEnqueueWriteBuffer(
            command_queue,
            D_d,
            blocking,
            0,
            nbytes_D,
            D_h,
            0,
            NULL,
            &io_event
        ) 
    );

    // Print out the time for the event
    h_get_event_time_ms(
        &io_event,
        "Uploading Buffer D",
        &nbytes_D
    );

    H_ERRCHK(
        clEnqueueWriteBuffer(
            command_queue,
            E_d,
            blocking,
            0,
            nbytes_E,
            E_h,
            0,
            NULL,
            &io_event
        ) 
    );


    // Print out the time for the event
    h_get_event_time_ms(
        &io_event,
        "Uploading Buffer E",
        &nbytes_E
    );
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 8, 8 };
    
    // Desired global_size
    const size_t global_size[]={ N1_F, N0_F };
    
    // Enlarge the global size so that 
    // an integer number of local sizes fits within it
    h_fit_global_size(global_size, 
                      local_size, 
                      work_dim
    );
    
    // Event for the kernel
    cl_event kernel_event;
    
    // Now enqueue the kernel
    H_ERRCHK(
        clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            work_dim,
            NULL,
            global_size,
            local_size,
            0,
            NULL,
            &kernel_event
        ) 
    );

    // Get profiling information on the kernel event
    h_get_event_time_ms(
        &kernel_event,
        "Kernel execution",
        NULL
    );

    // Copy F_d back to F_h
    H_ERRCHK(
        clEnqueueReadBuffer(
            command_queue,
            F_d,
            blocking,
            0,
            nbytes_F,
            F_h,
            1,
            &kernel_event,
            &io_event
        ) 
    );

    // Get profiling information on the IO event
    h_get_event_time_ms(
        &io_event,
        "Downloading Buffer F",
        &nbytes_F
    );

    // Check the answer against a known solution
    float* F_answer_h = (float*)calloc(nbytes_F, 1);

    // Compute the known solution
    m_hadamard(D_h, E_h, F_answer_h, N0_F, N1_F);

    // Print the maximum error between matrices
    float max_err = m_max_error(F_h, F_answer_h, N0_F, N1_F);

    // Write out the result to file
    h_write_binary(D_h, "array_D.dat", nbytes_D);
    h_write_binary(E_h, "array_E.dat", nbytes_E);
    h_write_binary(F_h, "array_F.dat", nbytes_F);

    H_ERRCHK(clReleaseMemObject(D_d));
    H_ERRCHK(clReleaseMemObject(E_d));
    H_ERRCHK(clReleaseMemObject(F_d));
    
    //// End code: ////

    // Clean up memory that was allocated on the host 
    free(D_h);
    free(E_h);
    free(F_h);
    free(F_answer_h);
    
    // Clean up command queues
    h_release_command_queues(
        command_queues, 
        num_command_queues
    );
    
    // Clean up devices, queues, and contexts
    h_release_devices(
        devices,
        num_devices,
        contexts,
        platforms
    );
}

