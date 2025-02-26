#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h> // Added for OpenMP support
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

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    coo_matrix global_coo;
    int global_num_rows, global_num_cols;
    float *x = NULL;

    if (rank == 0)
    {
        if (argc < 2)
        {
            printf("Give a MatrixMarket file.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        char *mm_filename = argv[1];
        printf("Reading matrix from file %s\n", mm_filename);
        read_coo_matrix(&global_coo, mm_filename);
        global_num_rows = global_coo.num_rows;
        global_num_cols = global_coo.num_cols;

        x = (float *)malloc(global_num_cols * sizeof(float));
        srand(13);
        for (int i = 0; i < global_num_cols; i++)
        {
            x[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
        }
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, global_num_rows, global_num_cols, global_coo.num_nonzeros);
        fflush(stdout);
    }

    MPI_Bcast(&global_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        x = (float *)malloc(global_num_cols * sizeof(float));
    }
    MPI_Bcast(x, global_num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int rows_per_proc = global_num_rows / size;
    int extra = global_num_rows % size;
    int rstart = rank * rows_per_proc + (rank < extra ? rank : extra);
    int rcount = rows_per_proc + (rank < extra ? 1 : 0);
    int rend = rstart + rcount;

    int local_nonzeros;
    int *local_rows = NULL;
    int *local_cols = NULL;
    float *local_vals = NULL;

    if (rank == 0)
    {
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
                local_rows[idx] = r - rstart;
                local_cols[idx] = global_coo.cols[i];
                local_vals[idx] = global_coo.vals[i];
                idx++;
            }
        }
        // For every other process, filter its part
        for (int p = 1; p < size; p++)
        {
            int prstart = p * rows_per_proc + (p < extra ? p : extra);
            int prcount = rows_per_proc + (p < extra ? 1 : 0);
            int prend = prstart + prcount;
            int count = 0;
            // Count nonzeros
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
        // Other ranks receive the number of nonzeros and then
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

    float *local_y = (float *)calloc(rcount, sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

#pragma omp parallel for default(none) shared(local_nonzeros, local_y, local_rows, local_vals, x, local_cols)
    for (int i = 0; i < local_nonzeros; i++)
    {
#pragma omp atomic
        local_y[local_rows[i]] += local_vals[i] * x[local_cols[i]];
    }

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

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    double elapsed = t_end - t_start;

    if (rank == 0)
    {
        float *sequential_y = (float *)calloc(global_num_rows, sizeof(float));
        for (int i = 0; i < global_coo.num_nonzeros; i++)
        {
            sequential_y[global_coo.rows[i]] += global_coo.vals[i] * x[global_coo.cols[i]];
        }

        printf("\nVerifying results based on integer portions\n");
        verify_integer_portion(sequential_y, global_y, global_num_rows);
        double total_flops = 2.0 * global_coo.num_nonzeros;
        double gflops = (total_flops / elapsed) / 1e9;
        printf("Single spMV run took %f seconds, achieving %f GFLOP/s\n", elapsed, gflops);

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