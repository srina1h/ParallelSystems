#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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
void usage(int argc, char **argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n");
}

double benchmark_coo_spmv(coo_matrix *coo, float *x, float *y)
{
    int num_nonzeros = coo->num_nonzeros;

    // Start the timer for one iteration.
    timer t;
    timer_start(&t);

// Perform one iteration of SpMV using OpenMP.
#pragma omp parallel for
    for (int i = 0; i < num_nonzeros; i++)
    {
#pragma omp atomic
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }

    // Measure the elapsed time in seconds.
    double sec = seconds_elapsed(&t);

    // Convert seconds to milliseconds for printing.
    double msec = sec * 1000.0;

    // Calculate GFLOP/s: each nonzero requires two flops (a multiply and an add).
    double GFLOPs = (sec == 0) ? 0 : (2.0 * (double)num_nonzeros / sec) / 1e9;

    printf("\tbenchmarking COO-SpMV (1 iteration): %8.4f ms ( %5.2f GFLOP/s)\n",
           msec, GFLOPs);

    return sec;
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
    {
        mm_filename = argv[1];
        printf("Reading matrix from file %s\n", mm_filename);
    }

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    // Fill matrix with random values for testing.
    srand(13);
    for (int i = 0; i < coo.num_nonzeros; i++)
    {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
    }

    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

    // Initialize host arrays.
    float *x = (float *)malloc(coo.num_cols * sizeof(float));
    float *y = (float *)malloc(coo.num_rows * sizeof(float));

    for (int i = 0; i < coo.num_cols; i++)
    {
        x[i] = rand() / (RAND_MAX + 1.0);
    }
    for (int i = 0; i < coo.num_rows; i++)
        y[i] = 0;

    /* Benchmarking */
    double coo_gflops = benchmark_coo_spmv(&coo, x, y);

    delete_coo_matrix(&coo);
    free(x);
    free(y);

    return 0;
}