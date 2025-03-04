#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void distribute_matrix_blocks(float *global_matrix, float *local_matrix, 
                              int rows, int cols, int grid_size, 
                              MPI_Comm comm_2d, MPI_Datatype block_type) {
  int coords[2], rank;
  MPI_Comm_rank(comm_2d, &rank);
  MPI_Cart_coords(comm_2d, rank, 2, coords);
  
  // Calculate local block dimensions
  int local_rows = rows / grid_size;
  int local_cols = cols / grid_size;
  
  // Root process distributes matrix blocks to all processes
  if (rank == 0) {
    // Send blocks to all processes (including itself)
    for (int i = 0; i < grid_size; i++) {
      for (int j = 0; j < grid_size; j++) {
        // Calculate destination coordinates and rank
        int dest_coords[2] = {i, j};
        int dest_rank;
        MPI_Cart_rank(comm_2d, dest_coords, &dest_rank);
        
        // Calculate the starting position of the block in global matrix
        int start_row = i * local_rows;
        int start_col = j * local_cols;
        
        // For rank 0, directly copy the data
        if (dest_rank == 0) {
          for (int r = 0; r < local_rows; r++) {
            for (int c = 0; c < local_cols; c++) {
              local_matrix[r * local_cols + c] = 
                global_matrix[(start_row + r) * cols + (start_col + c)];
            }
          }
        } else {
          // Create a temporary buffer for the block to send
          float *block_buffer = (float *)malloc(local_rows * local_cols * sizeof(float));
          
          // Copy the block from global matrix to buffer
          for (int r = 0; r < local_rows; r++) {
            for (int c = 0; c < local_cols; c++) {
              block_buffer[r * local_cols + c] = 
                global_matrix[(start_row + r) * cols + (start_col + c)];
            }
          }
          
          // Send the block to destination process
          MPI_Send(block_buffer, local_rows * local_cols, MPI_FLOAT, 
                   dest_rank, 0, comm_2d);
          
          free(block_buffer);
        }
      }
    }
  } else {
    // Receive block from the root process
    MPI_Recv(local_matrix, local_rows * local_cols, MPI_FLOAT, 
             0, 0, comm_2d, MPI_STATUS_IGNORE);
  }
}

void summa_stationary_a(int m, int n, int k, int nprocs, int rank) {
  // Grid setup
  int grid_size = (int)sqrt(nprocs);
  
  // Create 2D process grid
  int dims[2] = {grid_size, grid_size};
  int periods[2] = {0, 0}; // Non-periodic grid
  MPI_Comm comm_2d;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2d);
  
  // Get process coordinates
  int coords[2];
  MPI_Cart_coords(comm_2d, rank, 2, coords);
  
  // Create row and column communicators
  MPI_Comm row_comm, col_comm;
  int remain_dims[2];
  
  // Row communicator: fixed row index, varying column index
  remain_dims[0] = 0; // Don't keep row dimension
  remain_dims[1] = 1; // Keep column dimension
  MPI_Cart_sub(comm_2d, remain_dims, &row_comm);
  
  // Column communicator: varying row index, fixed column index
  remain_dims[0] = 1; // Keep row dimension
  remain_dims[1] = 0; // Don't keep column dimension
  MPI_Cart_sub(comm_2d, remain_dims, &col_comm);
  
  // Calculate local matrix dimensions
  int local_m = m / grid_size;
  int local_n = n / grid_size;
  int local_k = k / grid_size;
  
  // Allocate memory for local matrices
  float *local_A = (float *)malloc(local_m * local_k * sizeof(float));
  float *local_B = (float *)malloc(local_k * local_n * sizeof(float));
  float *local_C = (float *)calloc(local_m * local_n, sizeof(float));
  float *temp_A = (float *)malloc(local_m * local_k * sizeof(float));
  float *temp_B = (float *)malloc(local_k * local_n * sizeof(float));
  float *temp_C = (float *)calloc(local_m * local_n, sizeof(float));
  
  // Generate matrices on root process and distribute
  float *A = NULL, *B = NULL, *C = NULL;
  if (rank == 0) {
    A = generate_matrix_A(m, k, rank);
    B = generate_matrix_B(k, n, rank);
    C = (float *)calloc(m * n, sizeof(float));
  }
  
  // Create MPI datatypes for matrix blocks
  MPI_Datatype block_type_A, block_type_B;
  MPI_Type_vector(local_m, local_k, k, MPI_FLOAT, &block_type_A);
  MPI_Type_vector(local_k, local_n, n, MPI_FLOAT, &block_type_B);
  MPI_Type_commit(&block_type_A);
  MPI_Type_commit(&block_type_B);
  
  // Distribute initial matrix blocks
  distribute_matrix_blocks(A, local_A, m, k, grid_size, comm_2d, block_type_A);
  distribute_matrix_blocks(B, local_B, k, n, grid_size, comm_2d, block_type_B);
  
  // Initialize C to zero
  memset(local_C, 0, local_m * local_n * sizeof(float));
  
  // SUMMA computation with A stationary
  for (int l = 0; l < grid_size; l++) {
    // Calculate the broadcast coordinates for this iteration
    int A_bcast_coords = (coords[1] + grid_size - l) % grid_size;
    int B_bcast_coords = (coords[0] + grid_size - l) % grid_size;
    
    // Broadcast A along row
    if (coords[1] == A_bcast_coords) {
      // Copy my local A block to temp_A
      memcpy(temp_A, local_A, local_m * local_k * sizeof(float));
    }
    
    // Broadcast A from process with column coordinate A_bcast_coords in each row
    MPI_Bcast(temp_A, local_m * local_k, MPI_FLOAT, A_bcast_coords, row_comm);
    
    // Broadcast B along column
    if (coords[0] == B_bcast_coords) {
      // Copy my local B block to temp_B
      memcpy(temp_B, local_B, local_k * local_n * sizeof(float));
    }
    
    // Broadcast B from process with row coordinate B_bcast_coords in each column
    MPI_Bcast(temp_B, local_k * local_n, MPI_FLOAT, B_bcast_coords, col_comm);
    
    // Local matrix multiplication: temp_C = temp_A * temp_B
    memset(temp_C, 0, local_m * local_n * sizeof(float));
    matmul(temp_A, temp_B, temp_C, local_m, local_n, local_k);
    
    // Accumulate result into local_C
    for (int i = 0; i < local_m * local_n; i++) {
      local_C[i] += temp_C[i];
    }
  }
  
  // Gather results into global matrix C on root process
  if (rank == 0) {
    // Copy local result into the correct position of global C
    for (int i = 0; i < local_m; i++) {
      for (int j = 0; j < local_n; j++) {
        C[i * n + j] = local_C[i * local_n + j];
      }
    }
    
    // Receive results from other processes
    for (int r = 1; r < nprocs; r++) {
      int r_coords[2];
      MPI_Cart_coords(comm_2d, r, 2, r_coords);
      float *recv_C = (float *)malloc(local_m * local_n * sizeof(float));
      
      MPI_Recv(recv_C, local_m * local_n, MPI_FLOAT, r, 0, comm_2d, MPI_STATUS_IGNORE);
      
      // Copy received data to the correct position in global C
      int start_row = r_coords[0] * local_m;
      int start_col = r_coords[1] * local_n;
      
      for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < local_n; j++) {
          C[(start_row + i) * n + (start_col + j)] = recv_C[i * local_n + j];
        }
      }
      
      free(recv_C);
    }
    
    // Verify results
    verify_result(C, A, B, m, n, k);
    
    // Clean up global matrices
    free(A);
    free(B);
    free(C);
  } else {
    // Send local results to root process
    MPI_Send(local_C, local_m * local_n, MPI_FLOAT, 0, 0, comm_2d);
  }
  
  // Clean up
  free(local_A);
  free(local_B);
  free(local_C);
  free(temp_A);
  free(temp_B);
  free(temp_C);
  MPI_Type_free(&block_type_A);
  MPI_Type_free(&block_type_B);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&comm_2d);
}

void summa_stationary_b(int m, int n, int k, int nprocs, int rank) {
  // Grid setup
  int grid_size = (int)sqrt(nprocs);
  
  // Create 2D process grid
  int dims[2] = {grid_size, grid_size};
  int periods[2] = {0, 0}; // Non-periodic grid
  MPI_Comm comm_2d;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2d);
  
  // Get process coordinates
  int coords[2];
  MPI_Cart_coords(comm_2d, rank, 2, coords);
  
  // Create row and column communicators
  MPI_Comm row_comm, col_comm;
  int remain_dims[2];
  
  // Row communicator: fixed row index, varying column index
  remain_dims[0] = 0; // Don't keep row dimension
  remain_dims[1] = 1; // Keep column dimension
  MPI_Cart_sub(comm_2d, remain_dims, &row_comm);
  
  // Column communicator: varying row index, fixed column index
  remain_dims[0] = 1; // Keep row dimension
  remain_dims[1] = 0; // Don't keep column dimension
  MPI_Cart_sub(comm_2d, remain_dims, &col_comm);
  
  // Determine local block sizes
  int local_m = m / grid_size;
  int local_n = n / grid_size;
  int local_k = k / grid_size;
  
  // Generate random matrices on root process
  float *A = NULL, *B = NULL, *C = NULL;
  if (rank == 0) {
    A = generate_matrix_A(m, k, rank);
    B = generate_matrix_B(k, n, rank);
    C = (float *)calloc(m * n, sizeof(float));
  }
  
  // Allocate local matrices
  float *local_A = (float *)malloc(local_m * local_k * sizeof(float));
  float *local_B = (float *)malloc(local_k * local_n * sizeof(float));
  float *local_C = (float *)calloc(local_m * local_n, sizeof(float));
  float *temp_A = (float *)malloc(local_m * local_k * sizeof(float));
  float *temp_B = (float *)malloc(local_k * local_n * sizeof(float));
  float *temp_C = (float *)calloc(local_m * local_n, sizeof(float));
  
  // Create MPI datatypes for matrix blocks
  MPI_Datatype block_type_A, block_type_B;
  MPI_Type_vector(local_m, local_k, k, MPI_FLOAT, &block_type_A);
  MPI_Type_vector(local_k, local_n, n, MPI_FLOAT, &block_type_B);
  MPI_Type_commit(&block_type_A);
  MPI_Type_commit(&block_type_B);
  
  // Distribute matrix blocks
  distribute_matrix_blocks(A, local_A, m, k, grid_size, comm_2d, block_type_A);
  distribute_matrix_blocks(B, local_B, k, n, grid_size, comm_2d, block_type_B);
  
  // Initialize C to zero
  memset(local_C, 0, local_m * local_n * sizeof(float));
  
  // SUMMA computation with B stationary
  for (int l = 0; l < grid_size; l++) {
    // Calculate the broadcast coordinates for this iteration
    int A_bcast_coords = (coords[0] + grid_size - l) % grid_size;
    int B_bcast_coords = (coords[1] + grid_size - l) % grid_size;
    
    // Broadcast A along column
    if (coords[0] == A_bcast_coords) {
      // Copy my local A block to temp_A
      memcpy(temp_A, local_A, local_m * local_k * sizeof(float));
    }
    
    // Broadcast A from process with row coordinate A_bcast_coords in each column
    MPI_Bcast(temp_A, local_m * local_k, MPI_FLOAT, A_bcast_coords, col_comm);
    
    // Broadcast B along row
    if (coords[1] == B_bcast_coords) {
      // Copy my local B block to temp_B
      memcpy(temp_B, local_B, local_k * local_n * sizeof(float));
    }
    
    // Broadcast B from process with column coordinate B_bcast_coords in each row
    MPI_Bcast(temp_B, local_k * local_n, MPI_FLOAT, B_bcast_coords, row_comm);
    
    // Local matrix multiplication: temp_C = temp_A * temp_B
    memset(temp_C, 0, local_m * local_n * sizeof(float));
    matmul(temp_A, temp_B, temp_C, local_m, local_n, local_k);
    
    // Accumulate result into local_C
    for (int i = 0; i < local_m * local_n; i++) {
      local_C[i] += temp_C[i];
    }
  }
  
  // Gather results
  if (rank == 0) {
    // Copy local result into the correct position of global C
    for (int i = 0; i < local_m; i++) {
      for (int j = 0; j < local_n; j++) {
        C[i * n + j] = local_C[i * local_n + j];
      }
    }
    
    // Receive results from other processes
    for (int r = 1; r < nprocs; r++) {
      int r_coords[2];
      MPI_Cart_coords(comm_2d, r, 2, r_coords);
      float *recv_C = (float *)malloc(local_m * local_n * sizeof(float));
      
      MPI_Recv(recv_C, local_m * local_n, MPI_FLOAT, r, 0, comm_2d, MPI_STATUS_IGNORE);
      
      // Copy received data to the correct position in global C
      int start_row = r_coords[0] * local_m;
      int start_col = r_coords[1] * local_n;
      
      for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < local_n; j++) {
          C[(start_row + i) * n + (start_col + j)] = recv_C[i * local_n + j];
        }
      }
      
      free(recv_C);
    }
    
    // Verify results
    verify_result(C, A, B, m, n, k);
    
    // Clean up global matrices
    free(A);
    free(B);
    free(C);
  } else {
    // Send local results to root process
    MPI_Send(local_C, local_m * local_n, MPI_FLOAT, 0, 0, comm_2d);
  }
  
  // Clean up
  free(local_A);
  free(local_B);
  free(local_C);
  free(temp_A);
  free(temp_B);
  free(temp_C);
  MPI_Type_free(&block_type_A);
  MPI_Type_free(&block_type_B);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&comm_2d);
}

int main(int argc, char *argv[]) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  
  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Get the number of processes
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  SummaOpts opts;
  opts = parse_args(argc, argv);
  
  // Broadcast options to all processes
  MPI_Bcast(&opts, sizeof(SummaOpts), MPI_BYTE, 0, MPI_COMM_WORLD);
  
  // Check if number of processes is a perfect square
  int grid_size = (int)sqrt(nprocs);
  if (grid_size * grid_size != nprocs) {
    if (rank == 0) {
      printf("Error: Number of processes (%d) must be a perfect square\n", nprocs);
    }
    MPI_Finalize();
    return 1;
  }
  
  // Check if matrix dimensions are compatible with grid size
  if (opts.m % grid_size != 0 || opts.n % grid_size != 0 ||
      opts.k % grid_size != 0) {
    if (rank == 0) {
      printf("Error: Matrix dimensions must be divisible by grid size (%d)\n",
           grid_size);
    }
    MPI_Finalize();
    return 1;
  }
  
  if (rank == 0) {
    printf("\nMatrix Dimensions:\n");
    printf("A: %d x %d\n", opts.m, opts.k);
    printf("B: %d x %d\n", opts.k, opts.n);
    printf("C: %d x %d\n", opts.m, opts.n);
    printf("Grid size: %d x %d\n", grid_size, grid_size);
    printf("Block size: %d\n", opts.block_size);
    printf("Algorithm: Stationary %c\n", opts.stationary);
    printf("Verbose: %s\n", opts.verbose ? "true" : "false");
  }
  
  // Call the appropriate SUMMA function based on algorithm variant
  if (opts.stationary == 'A' || opts.stationary == 'a') {
    summa_stationary_a(opts.m, opts.n, opts.k, nprocs, rank);
  } else if (opts.stationary == 'B' || opts.stationary == 'b') {
    summa_stationary_b(opts.m, opts.n, opts.k, nprocs, rank);
  } else {
    if (rank == 0) {
      printf("Error: Unknown stationary option '%c'. Use 'A' or 'B'.\n",
           opts.stationary);
    }
    MPI_Finalize();
    return 1;
  }
  
  MPI_Finalize();
  return 0;
}