#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a, b) \
    ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })

#define min(a, b) \
    ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); \
   _a < _b ? _a : _b; })

void usage(int argc, char **argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be a real-valued sparse matrix in MatrixMarket format.\n");
}

void verify_integer_portion(float *sequential_y, float *parallel_y, int num_rows)
{
    int correct = 1;
    for (int i = 0; i < num_rows; i++)
    {
        int seq_int = (int)(sequential_y[i] + 0.5);
        int par_int = (int)(parallel_y[i] + 0.5);
        if (seq_int != par_int)
        {
            printf("Mismatch at index %d: sequential integer portion = %d, parallel integer portion = %d\n", i, seq_int, par_int);
            correct = 0;
        }
    }
    if (correct)
        printf("Verification successful: All integer portions match!\n");
    else
        printf("Verification failed: Some integer portions do not match!\n");
}

double benchmark_coo_spmv(coo_matrix *coo, float *x, float *y)
{
    int num_nonzeros = coo->num_nonzeros;
    // Make a copy of y to use for each iteration so that y doesn't accumulate.
    // Or, if you want to time the full accumulated operation, reinitialize y before verification.
    // For instance, use a local temporary result vector.
    float *temp_y = (float *)calloc(coo->num_rows, sizeof(float));

    // Warmup: perform one iteration using temp_y
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    for (int i = 0; i < num_nonzeros; i++)
    {
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }
    double estimated_time = seconds_elapsed(&time_one_iteration);

    // Determine number of iterations dynamically
    int num_iterations = MAX_ITER;
    if (estimated_time != 0)
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int)(TIME_LIMIT / estimated_time)));
    printf("\tPerforming %d iterations\n", num_iterations);

    // Zero out temp_y before timing multiple iterations.
    for (int i = 0; i < coo->num_rows; i++)
        temp_y[i] = 0;

    // Time several SpMV iterations on temp_y
    timer t;
    timer_start(&t);
    for (int j = 0; j < num_iterations; j++)
        for (int i = 0; i < num_nonzeros; i++)
        {
            temp_y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;

    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double)coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double)bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs);

    free(temp_y);
    return msec_per_iteration;
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    coo_matrix global_coo; // global matrix (only used on rank 0)
    int global_num_rows, global_num_cols;
    float *x = NULL; // global vector x

    if (rank == 0)
    {
        if (argc < 2)
        {
            printf("Give a MatrixMarket file.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        char *mm_filename = argv[1];
        read_coo_matrix(&global_coo, mm_filename);
        global_num_rows = global_coo.num_rows;
        global_num_cols = global_coo.num_cols;

        // Initialize global vector x with random values
        x = (float *)malloc(global_num_cols * sizeof(float));
        srand(13);
        for (int i = 0; i < global_num_cols; i++)
        {
            x[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
        }
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, global_num_rows, global_num_cols, global_coo.num_nonzeros);
        fflush(stdout);
    }

    // Broadcast matrix dimensions to all processes
    MPI_Bcast(&global_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Make sure all processes have vector x.
    if (rank != 0)
    {
        x = (float *)malloc(global_num_cols * sizeof(float));
    }
    MPI_Bcast(x, global_num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Determine the row partition for each MPI process.
    // Here, we partition rows as evenly as possible.
    int rows_per_proc = global_num_rows / size;
    int extra = global_num_rows % size;
    int rstart = rank * rows_per_proc + (rank < extra ? rank : extra);
    int rcount = rows_per_proc + (rank < extra ? 1 : 0);
    int rend = rstart + rcount;

    // Partition the matrix among processes.
    // Rank 0 filters the global COO matrix and sends each process its appropriate rows.
    int local_nonzeros;
    int *local_rows = NULL;
    int *local_cols = NULL;
    float *local_vals = NULL;

    if (rank == 0)
    {
        // Filter for rank 0 (its part: rows [rstart, rend))
        local_nonzeros = 0;
        for (int i = 0; i < global_coo.num_nonzeros; i++)
        {
            int r = global_coo.rows[i];
            if (r >= rstart && r < rend)
                local_nonzeros++;
        }
        local_rows = (int *)malloc(local_nonzeros * sizeof(int));
        local_cols = (int *)malloc(local_nonzeros * sizeof(int));
        local_vals = (float *)malloc(local_nonzeros * sizeof(float));
        int idx = 0;
        for (int i = 0; i < global_coo.num_nonzeros; i++)
        {
            int r = global_coo.rows[i];
            if (r >= rstart && r < rend)
            {
                // Adjust global to local row index
                local_rows[idx] = r - rstart;
                local_cols[idx] = global_coo.cols[i];
                local_vals[idx] = global_coo.vals[i];
                idx++;
            }
        }
        // For every other process, filter its part and send the data.
        for (int p = 1; p < size; p++)
        {
            int prstart = p * rows_per_proc + (p < extra ? p : extra);
            int prcount = rows_per_proc + (p < extra ? 1 : 0);
            int prend = prstart + prcount;
            int count = 0;
            // Count nonzeros for process p.
            for (int i = 0; i < global_coo.num_nonzeros; i++)
            {
                int r = global_coo.rows[i];
                if (r >= prstart && r < prend)
                    count++;
            }
            MPI_Send(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            if (count > 0)
            {
                int *tmp_rows = (int *)malloc(count * sizeof(int));
                int *tmp_cols = (int *)malloc(count * sizeof(int));
                float *tmp_vals = (float *)malloc(count * sizeof(float));
                int idx2 = 0;
                for (int i = 0; i < global_coo.num_nonzeros; i++)
                {
                    int r = global_coo.rows[i];
                    if (r >= prstart && r < prend)
                    {
                        tmp_rows[idx2] = r - prstart;
                        tmp_cols[idx2] = global_coo.cols[i];
                        tmp_vals[idx2] = global_coo.vals[i];
                        idx2++;
                    }
                }
                MPI_Send(tmp_rows, count, MPI_INT, p, 1, MPI_COMM_WORLD);
                MPI_Send(tmp_cols, count, MPI_INT, p, 2, MPI_COMM_WORLD);
                MPI_Send(tmp_vals, count, MPI_FLOAT, p, 3, MPI_COMM_WORLD);
                free(tmp_rows);
                free(tmp_cols);
                free(tmp_vals);
            }
        }
    }
    else
    {
        // Other ranks receive the number of nonzeros and then the data.
        MPI_Recv(&local_nonzeros, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (local_nonzeros > 0)
        {
            local_rows = (int *)malloc(local_nonzeros * sizeof(int));
            local_cols = (int *)malloc(local_nonzeros * sizeof(int));
            local_vals = (float *)malloc(local_nonzeros * sizeof(float));
            MPI_Recv(local_rows, local_nonzeros, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_cols, local_nonzeros, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_vals, local_nonzeros, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Build a local COO structure for the local partition.
    coo_matrix local_coo;
    local_coo.num_rows = rcount;
    local_coo.num_cols = global_num_cols;
    local_coo.num_nonzeros = local_nonzeros;
    local_coo.rows = local_rows;
    local_coo.cols = local_cols;
    local_coo.vals = local_vals;

    // Allocate local y vector (initialized to zero)
    float *local_y = (float *)calloc(rcount, sizeof(float));

    // Run the local SpMV benchmark computation.
    double local_time = benchmark_coo_spmv(&local_coo, x, local_y);

    // Gather the computed local y vectors back to rank 0.
    float *global_y = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        global_y = (float *)malloc(global_num_rows * sizeof(float));
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        for (int p = 0; p < size; p++)
        {
            int prcount = rows_per_proc + (p < extra ? 1 : 0);
            recvcounts[p] = prcount;
            int prstart = p * rows_per_proc + (p < extra ? p : extra);
            displs[p] = prstart;
        }
    }
    MPI_Gatherv(local_y, rcount, MPI_FLOAT, global_y, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        float *sequential_y = (float *)calloc(global_num_rows, sizeof(float));
        for (int i = 0; i < global_coo.num_nonzeros; i++)
        {
            sequential_y[global_coo.rows[i]] += global_coo.vals[i] * x[global_coo.cols[i]];
        }

        printf("\nVerifying results based on integer portions\n");
        verify_integer_portion(sequential_y, global_y, global_num_rows);
        printf("Parallel spMV complete. Global y computed.\n");
        free(global_y);
        free(recvcounts);
        free(displs);
        delete_coo_matrix(&global_coo);
        free(x);
    }
    free(local_y);
    if (rank != 0)
        free(x);
    if (local_rows)
        free(local_rows);
    if (local_cols)
        free(local_cols);
    if (local_vals)
        free(local_vals);

    MPI_Finalize();
    return 0;
}
