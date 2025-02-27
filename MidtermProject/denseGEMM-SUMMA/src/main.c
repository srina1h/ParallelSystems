#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to distribute matrix blocks among processes
void distribute_matrix_blocks(double *matrix, double *local_block, int rows, int cols, int block_rows, int block_cols, MPI_Comm grid_comm)
{
  // Scatter the matrix blocks to all processes
  int rank;
  MPI_Comm_rank(grid_comm, &rank);

  int grid_size;
  MPI_Comm_size(grid_comm, &grid_size);

  int p = sqrt(grid_size); // Process grid dimension

  // Create a derived datatype for the block
  MPI_Datatype block_type;
  MPI_Type_vector(block_rows, block_cols, cols, MPI_DOUBLE, &block_type);
  MPI_Type_commit(&block_type);

  if (rank == 0)
  {
    // Scatter blocks from root process
    for (int i = 0; i < p; i++)
    {
      for (int j = 0; j < p; j++)
      {
        int dest_rank = i * p + j;
        if (dest_rank == 0)
        {
          memcpy(local_block, matrix + i * block_rows * cols + j * block_cols, block_rows * block_cols * sizeof(double));
        }
        else
        {
          MPI_Send(matrix + i * block_rows * cols + j * block_cols, 1, block_type, dest_rank, 0, grid_comm);
        }
      }
    }
  }
  else
  {
    // Receive blocks in non-root processes
    MPI_Recv(local_block, block_rows * block_cols, MPI_DOUBLE, 0, 0, grid_comm, MPI_STATUS_IGNORE);
  }

  MPI_Type_free(&block_type);
}

// Stationary-A SUMMA implementation
void summa_stationary_a(int m, int n, int k, int nprocs, int rank)
{
  // Grid setup
  int p = sqrt(nprocs); // Process grid dimension

  // Create a Cartesian communicator
  MPI_Comm grid_comm;
  int dims[2] = {p, p};
  int periods[2] = {0, 0}; // Non-periodic grid
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

  // Get process coordinates in the grid
  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  // Create column communicators for B movement
  MPI_Comm col_comm;
  MPI_Comm_split(grid_comm, coords[1], coords[0], &col_comm);

  // Determine local block sizes
  int block_m = (m + p - 1) / p;
  int block_k = (k + p - 1) / p;
  int block_n = (n + p - 1) / p;

  // Allocate memory for local matrices
  double *A_local = malloc(block_m * block_k * sizeof(double));
  double *B_temp = malloc(block_k * block_n * sizeof(double));
  double *C_local = calloc(block_m * block_n, sizeof(double)); // Initialize C_local to zero

  if (rank == 0)
  {
    double *A = generate_random_matrix(m, k);
    double *B = generate_random_matrix(k, n);
    distribute_matrix_blocks(A, A_local, m, k, block_m, block_k, grid_comm);
    distribute_matrix_blocks(B, B_temp /* Temporary */, k /* Rows */, n /* Cols */, block_k /* Block rows */, block_n /* Block cols */, grid_comm);
    free(A);
    free(B);
  }
}

void summa_stationary_b(int m, int n, int k, int nprocs, int rank)
{
  // Determine the process grid dimension (p x p, where p = sqrt(nprocs)) [2]
  int p = (int)sqrt(nprocs);

  // Create a 2D Cartesian communicator with no periodicity [2]
  int dims[2] = {p, p};
  int periods[2] = {0, 0};
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

  // Get the coordinates of this process in the grid [2]
  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);
  int myRow = coords[0];
  int myCol = coords[1];

  // Create a row communicator so that within each row the A block can be broadcast [2]
  MPI_Comm row_comm;
  MPI_Comm_split(grid_comm, myRow, myCol, &row_comm);

  // Calculate local block sizes using ceiling division
  int block_m = (m + p - 1) / p;
  int block_k = (k + p - 1) / p;
  int block_n = (n + p - 1) / p;

  // Allocate local buffers. Each process maintains its fixed B block,
  // a temporary buffer for the broadcasted A block, and a local C block to accumulate results [3].
  double *A_local = (double *)malloc(block_m * block_k * sizeof(double));
  double *B_local = (double *)malloc(block_k * block_n * sizeof(double));
  double *A_temp = (double *)malloc(block_m * block_k * sizeof(double)); // used for broadcast
  double *C_local = (double *)calloc(block_m * block_n, sizeof(double)); // initialize C_local to zeros

  // Generate full matrices A and B on the root process [1]
  double *A = NULL;
  double *B = NULL;
  if (rank == 0)
  {
    A = generate_random_matrix(m, k);
    B = generate_random_matrix(k, n);
  }

  // Distribute matrix A among processes.
  // In Stationary-B, each process in row i gets an A-local block that will
  // be broadcasted later when its column coordinate equals the current iteration index [2]
  distribute_matrix_blocks(A, A_local, m, k, block_m, block_k, grid_comm);

  // Distribute matrix B. Each process retains its local B block which remains stationary [2]
  distribute_matrix_blocks(B, B_local, k, n, block_k, block_n, grid_comm);

  // Free full matrices on root after distribution.
  if (rank == 0)
  {
    free(A);
    free(B);
  }

  // Main SUMMA computation loop over the panel index (iterating over block columns in A)
  for (int iter = 0; iter < p; iter++)
  {
    // In each row, the process whose column coordinate equals iter is the root for current broadcast.
    // Its A_local block is used for this iteration.
    if (myCol == iter)
    {
      memcpy(A_temp, A_local, block_m * block_k * sizeof(double));
    }
    // Broadcast the A block along the row (all processes in the same row receive the block) [3]
    MPI_Bcast(A_temp, block_m * block_k, MPI_DOUBLE, iter, row_comm);

    // Compute the local matrix multiplication: C_local += A_temp * B_local
    matrix_multiply_add(A_temp, B_local, C_local, block_m, block_k, block_n);
  }

  // Gather the computed C blocks back to the root process.
  double *C = NULL;
  if (rank == 0)
  {
    C = (double *)malloc(m * n * sizeof(double));
  }
  // Here, we manually gather the blocks by having non-root processes send their results,
  // and the root process receives and places them in the proper locations.
  if (rank == 0)
  {
    // Copy the root's own block.
    for (int i = 0; i < block_m; i++)
    {
      memcpy(&C[i * n], &C_local[i * block_n], block_n * sizeof(double));
    }
    // Receive blocks from all other processes.
    for (int proc = 1; proc < nprocs; proc++)
    {
      int proc_coords[2];
      MPI_Cart_coords(grid_comm, proc, 2, proc_coords);
      int dest_row = proc_coords[0] * block_m;
      int dest_col = proc_coords[1] * block_n;
      double *temp_block = (double *)malloc(block_m * block_n * sizeof(double));
      MPI_Recv(temp_block, block_m * block_n, MPI_DOUBLE, proc, 0, grid_comm, MPI_STATUS_IGNORE);
      for (int i = 0; i < block_m; i++)
      {
        memcpy(&C[(dest_row + i) * n + dest_col], &temp_block[i * block_n], block_n * sizeof(double));
      }
      free(temp_block);
    }
  }
  else
  {
    MPI_Send(C_local, block_m * block_n, MPI_DOUBLE, 0, 0, grid_comm);
  }

  // Optionally, one can verify or print part of the final C matrix on the root process [3]

  // Clean up allocated memory and communicators.
  free(A_local);
  free(B_local);
  free(A_temp);
  free(C_local);
  if (rank == 0)
  {
    free(C);
  }
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&grid_comm);
}

int main(int argc, char *argv[])
{
  int rank, nprocs;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the total number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  SummaOpts opts;
  // Parse command-line arguments on the root process
  if (rank == 0)
  {
    opts = parse_args(argc, argv);
  }

  // Broadcast options to all processes (field-by-field to ensure portability)
  MPI_Bcast(&(opts.m), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(opts.n), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(opts.k), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(opts.block_size), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(opts.stationary), 1, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(opts.verbose), 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Check if the number of processes forms a perfect square grid
  int grid_size = (int)sqrt((double)nprocs);
  if (grid_size * grid_size != nprocs)
  {
    if (rank == 0)
    {
      printf("Error: Number of processes must form a perfect square grid.\n");
    }
    MPI_Finalize();
    return 1;
  }

  // Verify matrix dimensions are compatible with the grid size
  if (opts.m % grid_size != 0 || opts.n % grid_size != 0 || opts.k % grid_size != 0)
  {
    if (rank == 0)
    {
      printf("Error: Matrix dimensions must be divisible by grid size (%d).\n", grid_size);
    }
    MPI_Finalize();
    return 1;
  }

  // Print configuration details (only from the root process)
  if (rank == 0)
  {
    printf("\nMatrix Dimensions:\n");
    printf("A: %d x %d\n", opts.m, opts.k);
    printf("B: %d x %d\n", opts.k, opts.n);
    printf("C: %d x %d\n", opts.m, opts.n);
    printf("Grid size: %d x %d\n", grid_size, grid_size);
    printf("Block size: %d\n", opts.block_size);
    printf("Algorithm: Stationary %c\n", opts.stationary);
    printf("Verbose: %s\n", opts.verbose ? "true" : "false");
  }

  // Call the appropriate SUMMA function based on the stationary option
  if (opts.stationary == 'A')
  {
    summa_stationary_a(opts.m, opts.n, opts.k, nprocs, rank);
  }
  else if (opts.stationary == 'B')
  {
    summa_stationary_b(opts.m, opts.n, opts.k, nprocs, rank);
  }
  else
  {
    if (rank == 0)
    {
      printf("Error: Unknown stationary option '%c'. Use 'A' or 'B'.\n", opts.stationary);
    }
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}