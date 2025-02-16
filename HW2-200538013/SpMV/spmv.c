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

    // Warmup
    timer time_one_iteration;
    timer_start(&time_one_iteration);

// Parallelized warmup loop with OpenMP.
// Atomic directive is used to protect the update into y.
#pragma omp parallel for
    for (int i = 0; i < num_nonzeros; i++)
    {
#pragma omp atomic
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }
    double estimated_time = seconds_elapsed(&time_one_iteration);
    // Determine number of iterations dynamically (using MIN_ITER, MAX_ITER, TIME_LIMIT)
    int num_iterations = MAX_ITER;
    if (estimated_time != 0)
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int)(TIME_LIMIT / estimated_time)));
    printf("\tPerforming %d iterations\n", num_iterations);

    // Timing several SpMV iterations
    timer t;
    timer_start(&t);
    for (int j = 0; j < num_iterations; j++)
    {
// Parallelized inner loop with OpenMP and atomic update for each nonzero.
#pragma omp parallel for
        for (int i = 0; i < num_nonzeros; i++)
        {
#pragma omp atomic
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    }
    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double)coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double)bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs);

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
    {
        mm_filename = argv;
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
#ifdef TESTING
    // Optionally write the matrix in COO format for testing.
    printf("Writing matrix in COO format to test_COO ...");
    FILE *fp = fopen("test_COO", "w");
    fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fprintf(fp, "coo.rows:\n");
    for (int i = 0; i < coo.num_nonzeros; i++)
    {
        fprintf(fp, "%d ", coo.rows[i]);
    }
    fprintf(fp, "\n\ncoo.cols:\n");
    for (int i = 0; i < coo.num_nonzeros; i++)
    {
        fprintf(fp, "%d ", coo.cols[i]);
    }
    fprintf(fp, "\n\ncoo.vals:\n");
    for (int i = 0; i < coo.num_nonzeros; i++)
    {
        fprintf(fp, "%f ", coo.vals[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("... done!\n");
#endif

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
#ifdef TESTING
    // Optionally write the x and y vectors for testing.
    printf("Writing x and y vectors ...");
    FILE *fp = fopen("test_x", "w");
    for (int i = 0; i < coo.num_cols; i++)
    {
        fprintf(fp, "%f\n", x[i]);
    }
    fclose(fp);
    fp = fopen("test_y", "w");
    for (int i = 0; i < coo.num_rows; i++)
    {
        fprintf(fp, "%f\n", y[i]);
    }
    fclose(fp);
    printf("... done!\n");
#endif

    delete_coo_matrix(&coo);
    free(x);
    free(y);

    return 0;
}