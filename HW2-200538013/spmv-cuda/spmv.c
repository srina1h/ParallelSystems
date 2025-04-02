#include <stdio.h>
#include <cuda_runtime.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a, b) \
    ({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

// Error checking macro
#define cudaCheckError()                                         \
    {                                                            \
        cudaError_t e = cudaGetLastError();                      \
        if (e != cudaSuccess)                                    \
        {                                                        \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                       \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

void usage(int argc, char **argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n");
}

// CUDA kernel for COO SpMV
__global__ void coo_spmv_kernel(int num_nonzeros, int *rows, int *cols,
                                float *vals, float *x, float *y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_nonzeros)
    {
        atomicAdd(&y[rows[i]], vals[i] * x[cols[i]]);
    }
}

// Separate kernel with segmented reduction could be more efficient for large matrices
// but this simple atomic approach works for most cases

double benchmark_cuda_coo_spmv(coo_matrix *coo, float *x, float *y)
{
    int num_nonzeros = coo->num_nonzeros;
    int num_rows = coo->num_rows;
    int num_cols = coo->num_cols;

    // Allocate device memory
    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;

    cudaMalloc((void **)&d_rows, num_nonzeros * sizeof(int));
    cudaMalloc((void **)&d_cols, num_nonzeros * sizeof(int));
    cudaMalloc((void **)&d_vals, num_nonzeros * sizeof(float));
    cudaMalloc((void **)&d_x, num_cols * sizeof(float));
    cudaMalloc((void **)&d_y, num_rows * sizeof(float));
    cudaCheckError();

    // Copy data to device
    cudaMemcpy(d_rows, coo->rows, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, coo->cols, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, coo->vals, num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Configure kernel
    int blockSize = 256;
    int gridSize = (num_nonzeros + blockSize - 1) / blockSize;

    // Warmup
    cudaMemset(d_y, 0, num_rows * sizeof(float));
    timer time_one_iteration;
    timer_start(&time_one_iteration);

    coo_spmv_kernel<<<gridSize, blockSize>>>(num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y);
    cudaDeviceSynchronize();
    cudaCheckError();

    double estimated_time = seconds_elapsed(&time_one_iteration);

    // Determine # of iterations dynamically
    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else
    {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int)(TIME_LIMIT / estimated_time)));
    }
    printf("\tPerforming %d iterations\n", num_iterations);

    // Time several SpMV iterations
    timer t;
    timer_start(&t);

    for (int j = 0; j < num_iterations; j++)
    {
        cudaMemset(d_y, 0, num_rows * sizeof(float));
        coo_spmv_kernel<<<gridSize, blockSize>>>(num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y);
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // Copy result back
    cudaMemcpy(y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double)coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double)bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tGPU COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs);

    // Free device memory
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_x);
    cudaFree(d_y);

    return msec_per_iteration;
}

// Original CPU implementation for comparison
double benchmark_cpu_coo_spmv(coo_matrix *coo, float *x, float *y)
{
    int num_nonzeros = coo->num_nonzeros;

    // warmup
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    for (int i = 0; i < num_nonzeros; i++)
    {
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }

    double estimated_time = seconds_elapsed(&time_one_iteration);

    // determine # of seconds dynamically
    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else
    {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int)(TIME_LIMIT / estimated_time)));
    }
    printf("\tPerforming %d iterations\n", num_iterations);

    // Clear y for actual benchmark
    for (int i = 0; i < coo->num_rows; i++)
        y[i] = 0;

    // time several SpMV iterations
    timer t;
    timer_start(&t);
    for (int j = 0; j < num_iterations; j++)
    {
        for (int i = 0; i < coo->num_rows; i++)
            y[i] = 0;

        for (int i = 0; i < num_nonzeros; i++)
        {
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    }
    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double)coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double)bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tCPU COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs);

    return msec_per_iteration;
}

int main(int argc, char **argv)
{
    if (get_arg(argc, argv, "help") != NULL)
    {
        usage(argc, argv);
        return 0;
    }

    char *mm_filename = NULL;
    if (argc == 1)
    {
        printf("Give a MatrixMarket file.\n");
        return -1;
    }
    else
        mm_filename = argv[1];

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    // fill matrix with random values
    srand(13);
    for (int i = 0; i < coo.num_nonzeros; i++)
    {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
    }

    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

    // initialize host arrays
    float *x = (float *)malloc(coo.num_cols * sizeof(float));
    float *y_cpu = (float *)malloc(coo.num_rows * sizeof(float));
    float *y_gpu = (float *)malloc(coo.num_rows * sizeof(float));

    for (int i = 0; i < coo.num_cols; i++)
    {
        x[i] = rand() / (RAND_MAX + 1.0);
    }

    for (int i = 0; i < coo.num_rows; i++)
    {
        y_cpu[i] = 0;
        y_gpu[i] = 0;
    }

    // Run CPU benchmark
    printf("\n--- CPU Benchmark ---\n");
    double cpu_time = benchmark_cpu_coo_spmv(&coo, x, y_cpu);

    // Run GPU benchmark
    printf("\n--- GPU Benchmark ---\n");
    double gpu_time = benchmark_cuda_coo_spmv(&coo, x, y_gpu);

    // Validate results
    printf("\n--- Validation ---\n");
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (int i = 0; i < coo.num_rows; i++)
    {
        float diff = fabs(y_cpu[i] - y_gpu[i]);
        if (diff > max_diff)
            max_diff = diff;

        float rel_diff = (fabs(y_cpu[i]) > 1e-6) ? diff / fabs(y_cpu[i]) : diff;
        if (rel_diff > max_rel_diff)
            max_rel_diff = rel_diff;
    }
    printf("Max absolute difference: %e\n", max_diff);
    printf("Max relative difference: %e\n", max_rel_diff);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);

    delete_coo_matrix(&coo);
    free(x);
    free(y_cpu);
    free(y_gpu);

    return 0;
}