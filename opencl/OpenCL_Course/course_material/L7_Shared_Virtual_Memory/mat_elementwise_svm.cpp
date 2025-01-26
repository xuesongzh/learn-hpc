/* Code to perform Hadamard (elementwise) multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

// Define the size of the arrays to be computed
#define NROWS_F 8
#define NCOLS_F 4

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

    //// Step 2. Discover resources ////
    
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

    //// Step 3. Allocate command queues and choose a compute device ////
    
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
    // Also make sure command line arguments are sane
    assert(dev_index < num_devices);
    cl_context context = contexts[dev_index];
    cl_command_queue command_queue = command_queues[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on the device in use
    h_report_on_device(device);
    
    //// Check here if device supports fine-grained SVM

    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // D, E, F are of size (N0_F, N1_F)
    cl_uint N0_F = NROWS_F, N1_F = NCOLS_F;
    
    //// Step 4. Prepare matrices D, E, and F on the Host ////

    // Number of bytes in each array
    size_t nbytes_D=N0_F*N1_F*sizeof(float);   
    size_t nbytes_E=N0_F*N1_F*sizeof(float);
    size_t nbytes_F=N0_F*N1_F*sizeof(float);

    // Allocate memory for matrices A, B, and C on the host
    cl_float* D_h = (cl_float*)h_alloc(nbytes_D);
    cl_float* E_h = (cl_float*)h_alloc(nbytes_E);
    cl_float* F_h = (cl_float*)h_alloc(nbytes_F);

    // Fill A_h and B_h with random numbers 
    // using the matrix helper library
    m_random(D_h, N0_F, N1_F);
    m_random(E_h, N0_F, N1_F);
    
    //// Step 5. Allocate OpenCL Buffers for matrices D, E, and F ////

    // Make ordinary OpenCL buffers for matrices D, E, and F
    cl_mem D_d = clCreateBuffer(
            context, 
            CL_MEM_READ_WRITE, 
            nbytes_D, 
            NULL, 
            &errcode
    );
    H_ERRCHK(errcode);
    
    cl_mem E_d = clCreateBuffer(
            context, 
            CL_MEM_READ_WRITE, 
            nbytes_E, 
            NULL, 
            &errcode
    );
    H_ERRCHK(errcode);
    
    cl_mem F_d = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_F, 
                                     NULL, 
                                     &errcode);
    H_ERRCHK(errcode);


    //// Step 6. Build the program from source for the chosen compute device ////
    
    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_elementwise.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);

    //// Step 7. Create a kernel from the compiled program and set arguments ////
    
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_elementwise", &errcode);
    H_ERRCHK(errcode);
    
    // Set arguments to the kernel (not thread safe)
    H_ERRCHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &D_d));
    H_ERRCHK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &E_d));
    H_ERRCHK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &F_d));
    H_ERRCHK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &N0_F));
    H_ERRCHK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &N1_F));

    //// Step 8. Upload matrices D and E from the host to the OpenCL device Buffers ////

    // Write memory from D_h and E_h on the host 
    // to D_d and E_d on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;

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
            NULL
        )
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
            NULL
        )
    );
    
    //// Step 9. Run the kernel to compute F from D and E ////
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 2, 2 };
    
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

    // Wait on the kernel to finish
    H_ERRCHK(clWaitForEvents(1, &kernel_event));
   
    //// Step 10. Copy the Buffer for matrix C back to the host ////

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
            NULL
        ) 
    );

    //// Step 11. Test the answer against a known solution
    //// And write the contents of the matrices out to disk

    // Check the answer against a known solution
    float* F_answer_h = (float*)calloc(nbytes_F, 1);
    float* F_residual_h = (float*)calloc(nbytes_F, 1);

    // Compute the known solution
    m_hadamard(D_h, E_h, F_answer_h, N0_F, N1_F);

    // Compute the residual between F_h and F_answer_h
    m_residual(F_answer_h, F_h, F_residual_h, N0_F, N1_F);

    // Pretty print the output matrices
    std::cout << "The output array F_h (as computed with OpenCL) is\n";
    m_show_matrix(F_h, N0_F, N1_F);

    std::cout << "The CPU solution (F_answer_h) is \n";
    m_show_matrix(F_answer_h, N0_F, N1_F);
    
    std::cout << "The residual (F_answer_h-F_h) is\n";
    m_show_matrix(F_residual_h, N0_F, N1_F);

    // Print the maximum error between matrices
    float max_err = m_max_error(F_h, F_answer_h, N0_F, N1_F);

    // Write out the arrays to file
    h_write_binary(D_h, "array_D.dat", nbytes_D);
    h_write_binary(E_h, "array_E.dat", nbytes_E);
    h_write_binary(F_h, "array_F.dat", nbytes_F);

    //// Step 12. Clean up arrays and release resources
    
    // Free the OpenCL buffers
    H_ERRCHK(clReleaseMemObject(D_d));
    H_ERRCHK(clReleaseMemObject(E_d));
    H_ERRCHK(clReleaseMemObject(F_d));
    
    // Clean up memory that was allocated on the host 
    free(D_h);
    free(E_h);
    free(F_h);
    free(F_answer_h);
    free(F_residual_h);
    
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

