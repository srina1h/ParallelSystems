#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Structure to hold timing information
typedef struct {
    double init_time;
    double dist_time;
    double comp_time;
    double gather_time;
    double total_time;
} TimingInfo;

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

void print_timing_info(TimingInfo *timing, char variant, int rank) {
    if (rank == 0) {
        printf("\nTiming for SUMMA Stationary %c:\n", variant);
        printf("Initialization time: %f seconds\n", timing->init_time);
        printf("Distribution time:   %f seconds\n", timing->dist_time);
        printf("Computation time:    %f seconds\n", timing->comp_time);
        printf("Gathering time:      %f seconds\n", timing->gather_time);
        printf("Total time:          %f seconds\n", timing->total_time);
        printf("------------------------------------\n");
    }
}

void summa_stationary_a(int m, int n, int k, int nprocs, int rank) {
    TimingInfo timing = {0};
    double start_time, end_time;
    
    // Start total timing
    start_time = MPI_Wtime();
    
    // Start initialization timing
    double init_start = MPI_Wtime();
    
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
    int my_row = coords[0];
    int my_col = coords[1];
    
    // 1. Create row and column communicators
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
    
    // 2. Calculate local block dimensions
    int block_m = (m + grid_size - 1) / grid_size; // ceiling division
    int block_k = (k + grid_size - 1) / grid_size;
    int block_n = (n + grid_size - 1) / grid_size;
    
    // Calculate actual local dimensions (handle edge cases)
    int local_m = (my_row == grid_size - 1) ? (m - my_row * block_m) : block_m;
    int local_k = (my_col == grid_size - 1) ? (k - my_col * block_k) : block_k;
    int local_n = (my_col == grid_size - 1) ? (n - my_col * block_n) : block_n;
    
    // For simplicity, we'll assume dimensions are divisible by grid_size
    local_m = m / grid_size;
    local_k = k / grid_size;
    local_n = n / grid_size;
    
    // 3. Allocate memory for local matrices
    float *local_A = (float *)malloc(local_m * local_k * sizeof(float));  // Fixed A_ij
    float *B_temp = (float *)malloc(local_k * local_n * sizeof(float));  // Temporary B
    float *C_temp = (float *)calloc(local_m * local_n, sizeof(float));   // Partial results
    float *local_C = (float *)calloc(local_m * local_n, sizeof(float));  // Final local results
    
    // Generate matrices on root process
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
    
    timing.init_time = MPI_Wtime() - init_start;
    
    // Start distribution timing
    double dist_start = MPI_Wtime();
    
    // Distribute initial matrix blocks
    distribute_matrix_blocks(A, local_A, m, k, grid_size, comm_2d, block_type_A);
    distribute_matrix_blocks(B, B_temp, k, n, grid_size, comm_2d, block_type_B);
    
    timing.dist_time = MPI_Wtime() - dist_start;
    
    // Start computation timing
    double comp_start = MPI_Wtime();
    
    // Initialize C_temp to zero
    memset(C_temp, 0, local_m * local_n * sizeof(float));
    
    // 4. Main computation loop - only over columns (j = 0 to p-1)
    for (int j = 0; j < grid_size; j++) {
        // Save current B block if needed for broadcast
        float *B_to_broadcast = (float *)malloc(local_k * local_n * sizeof(float));
        if (my_col == j) {
            memcpy(B_to_broadcast, B_temp, local_k * local_n * sizeof(float));
        }
        
        // 5. Broadcast B within each column from process(k,j) to all processes in column j
        // Find the process in current column j with row id = j (that's the kth process)
        int source_row = j % grid_size;
        int source_coords[2] = {source_row, j};
        int source_rank;
        MPI_Cart_rank(comm_2d, source_coords, &source_rank);
        
        // Only processes in column j need to broadcast
        if (my_col == j) {
            MPI_Bcast(B_to_broadcast, local_k * local_n, MPI_FLOAT, source_row, col_comm);
        }
        
        // If this process is in column j, use the broadcasted B
        if (my_col == j) {
            // 6. Local computation: C_temp += A_ij × B_temp
            // Since A is stationary, we multiply our fixed A with the received B
            float *temp_result = (float *)calloc(local_m * local_n, sizeof(float));
            matmul(local_A, B_to_broadcast, temp_result, local_m, local_n, local_k);
            
            // Accumulate result
            for (int i = 0; i < local_m * local_n; i++) {
                C_temp[i] += temp_result[i];
            }
            
            free(temp_result);
        }
        
        free(B_to_broadcast);
    }
    
    // 7. Perform reduction to collect C results - reduce-scatter within each row
    // For simplicity, we'll just gather all C_temp values to local_C
    // In a true implementation, this would be a reduce-scatter operation
    
    // Create a receive buffer for the reduction
    float *recv_buffer = (float *)calloc(local_m * local_n, sizeof(float));
    
    // Reduce all C_temp values within the row to get the final C blocks
    MPI_Reduce(C_temp, recv_buffer, local_m * local_n, MPI_FLOAT, MPI_SUM, 0, row_comm);
    
    // Copy results to local_C
    if (my_col == 0) {  // Only the first process in each row gets the result
        memcpy(local_C, recv_buffer, local_m * local_n * sizeof(float));
    }
    
    free(recv_buffer);
    
    // 8. Synchronize
    MPI_Barrier(comm_2d);
    
    timing.comp_time = MPI_Wtime() - comp_start;
    
    // Start gathering timing
    double gather_start = MPI_Wtime();
    
    // Gather results from all processes to construct the final C matrix
    if (rank == 0) {
        // Copy local results to the appropriate position in the global C matrix
        for (int i = 0; i < local_m; i++) {
            for (int j = 0; j < local_n; j++) {
                C[i * n + j] = local_C[i * local_n + j];
            }
        }
        
        // Gather results from other processes
        for (int r = 1; r < nprocs; r++) {
            int r_coords[2];
            MPI_Cart_coords(comm_2d, r, 2, r_coords);
            
            // Only gather from processes in column 0 (they have the final results)
            if (r_coords[1] == 0) {
                float *recv_C = (float *)malloc(local_m * local_n * sizeof(float));
                
                MPI_Recv(recv_C, local_m * local_n, MPI_FLOAT, r, 0, comm_2d, MPI_STATUS_IGNORE);
                
                int start_row = r_coords[0] * local_m;
                int start_col = 0; // Always 0 for column 0 processes
                
                for (int i = 0; i < local_m; i++) {
                    for (int j = 0; j < local_n; j++) {
                        C[(start_row + i) * n + (start_col + j)] = recv_C[i * local_n + j];
                    }
                }
                
                free(recv_C);
            }
        }
        
        // Verify results
        verify_result(C, A, B, m, n, k);
        
        // Clean up global matrices
        free(A);
        free(B);
        free(C);
    } else if (my_col == 0) {
        // Only processes in column 0 need to send their results
        MPI_Send(local_C, local_m * local_n, MPI_FLOAT, 0, 0, comm_2d);
    }
    
    timing.gather_time = MPI_Wtime() - gather_start;
    
    // Calculate total time
    timing.total_time = MPI_Wtime() - start_time;
    
    // Print timing information
    print_timing_info(&timing, 'A', rank);
    
    // Clean up
    free(local_A);
    free(B_temp);
    free(C_temp);
    free(local_C);
    MPI_Type_free(&block_type_A);
    MPI_Type_free(&block_type_B);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&comm_2d);
}

void summa_stationary_c(int m, int n, int k, int nprocs, int rank) {
    TimingInfo timing = {0};
    double start_time, end_time;
    
    // Start total timing
    start_time = MPI_Wtime();
    
    // Start initialization timing
    double init_start = MPI_Wtime();
    
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
    int my_row = coords[0];
    int my_col = coords[1];
    
    // Create row and column communicators
    MPI_Comm row_comm, col_comm;
    int remain_dims[2];
    
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(comm_2d, remain_dims, &row_comm);
    
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(comm_2d, remain_dims, &col_comm);
    
    // Calculate local matrix dimensions
    int local_m = m / grid_size;
    int local_n = n / grid_size;
    int local_k = k / grid_size;
    
    // Generate matrices on root process
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
    
    // Create MPI datatypes
    MPI_Datatype block_type_A, block_type_B;
    MPI_Type_vector(local_m, local_k, k, MPI_FLOAT, &block_type_A);
    MPI_Type_vector(local_k, local_n, n, MPI_FLOAT, &block_type_B);
    MPI_Type_commit(&block_type_A);
    MPI_Type_commit(&block_type_B);
    
    timing.init_time = MPI_Wtime() - init_start;
    
    // Start distribution timing
    double dist_start = MPI_Wtime();
    
    // Distribute initial matrix blocks
    distribute_matrix_blocks(A, local_A, m, k, grid_size, comm_2d, block_type_A);
    distribute_matrix_blocks(B, local_B, k, n, grid_size, comm_2d, block_type_B);
    
    timing.dist_time = MPI_Wtime() - dist_start;
    
    // Start computation timing
    double comp_start = MPI_Wtime();
    
    // Initialize C to zero
    memset(local_C, 0, local_m * local_n * sizeof(float));
    
    // SUMMA computation with C stationary
    for (int l = 0; l < grid_size; l++) {
        if (my_col == l) {
            memcpy(temp_A, local_A, local_m * local_k * sizeof(float));
        }
        
        MPI_Bcast(temp_A, local_m * local_k, MPI_FLOAT, l, row_comm);
        
        if (my_row == l) {
            memcpy(temp_B, local_B, local_k * local_n * sizeof(float));
        }
        
        MPI_Bcast(temp_B, local_k * local_n, MPI_FLOAT, l, col_comm);
        
        memset(temp_C, 0, local_m * local_n * sizeof(float));
        matmul(temp_A, temp_B, temp_C, local_m, local_n, local_k);
        
        for (int i = 0; i < local_m * local_n; i++) {
            local_C[i] += temp_C[i];
        }
    }
    
    timing.comp_time = MPI_Wtime() - comp_start;
    
    // Start gathering timing
    double gather_start = MPI_Wtime();
    
    // Gather results
    if (rank == 0) {
        for (int i = 0; i < local_m; i++) {
            for (int j = 0; j < local_n; j++) {
                C[i * n + j] = local_C[i * local_n + j];
            }
        }
        
        for (int r = 1; r < nprocs; r++) {
            int r_coords[2];
            MPI_Cart_coords(comm_2d, r, 2, r_coords);
            float *recv_C = (float *)malloc(local_m * local_n * sizeof(float));
            
            MPI_Recv(recv_C, local_m * local_n, MPI_FLOAT, r, 0, comm_2d, MPI_STATUS_IGNORE);
            
            int start_row = r_coords[0] * local_m;
            int start_col = r_coords[1] * local_n;
            
            for (int i = 0; i < local_m; i++) {
                for (int j = 0; j < local_n; j++) {
                    C[(start_row + i) * n + (start_col + j)] = recv_C[i * local_n + j];
                }
            }
            
            free(recv_C);
        }
        
        verify_result(C, A, B, m, n, k);
        
        free(A);
        free(B);
        free(C);
    } else {
        MPI_Send(local_C, local_m * local_n, MPI_FLOAT, 0, 0, comm_2d);
    }
    
    timing.gather_time = MPI_Wtime() - gather_start;
    
    // Calculate total time
    timing.total_time = MPI_Wtime() - start_time;
    
    // Print timing information
    print_timing_info(&timing, 'C', rank);
    
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
  } else if (opts.stationary == 'C' || opts.stationary == 'c') {
    summa_stationary_c(opts.m, opts.n, opts.k, nprocs, rank);
  } else {
    if (rank == 0) {
      printf("Error: Unknown stationary option '%c'. Use 'A' or 'C'.\n",
           opts.stationary);
    }
    MPI_Finalize();
    return 1;
  }
  
  MPI_Finalize();
  return 0;
}