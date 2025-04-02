#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double init_time;
    double dist_time;
    double comp_time;
    double gather_time;
    double total_time;
    // Message counts
    int send_count;
    int recv_count;
    // Data movement
    size_t send_bytes;
    size_t recv_bytes;
} TimingInfo;

// Structure to track message counts during distribution
typedef struct {
    int send_count;
    int recv_count;
    size_t send_bytes;
    size_t recv_bytes;
} DistributionCounts;

void distribute_matrix_blocks(float *global_matrix, float *local_matrix, 
                              int rows, int cols, int grid_size, 
                              MPI_Comm comm_2d, MPI_Datatype block_type,
                              DistributionCounts *counts) {
  int coords[2], rank;
  MPI_Comm_rank(comm_2d, &rank);
  MPI_Cart_coords(comm_2d, rank, 2, coords);
  
  // Calculate local block dimensions
  int local_rows = rows / grid_size;
  int local_cols = cols / grid_size;
  
  if (rank == 0) {
    // Send blocks to all processes (including itself)
    for (int i = 0; i < grid_size; i++) {
      for (int j = 0; j < grid_size; j++) {
        // destination coordinates and rank
        int dest_coords[2] = {i, j};
        int dest_rank;
        MPI_Cart_rank(comm_2d, dest_coords, &dest_rank);
        
        // Calculate the starting position of the block in global matrix
        int start_row = i * local_rows;
        int start_col = j * local_cols;
        
        // For rank 0, copy data
        if (dest_rank == 0) {
          for (int r = 0; r < local_rows; r++) {
            for (int c = 0; c < local_cols; c++) {
              local_matrix[r * local_cols + c] = 
                global_matrix[(start_row + r) * cols + (start_col + c)];
            }
          }
        } else {
          // Create temporary buffer
          float *block_buffer = (float *)malloc(local_rows * local_cols * sizeof(float));
          
          // Copy the block from global matrix to buffer
          for (int r = 0; r < local_rows; r++) {
            for (int c = 0; c < local_cols; c++) {
              block_buffer[r * local_cols + c] = 
                global_matrix[(start_row + r) * cols + (start_col + c)];
            }
          }
          
          // Send the block to destination
          MPI_Send(block_buffer, local_rows * local_cols, MPI_FLOAT, 
                   dest_rank, 0, comm_2d);
          
          // Tracking data movement and messages
          if (counts != NULL) {
            counts->send_count++;
            counts->send_bytes += local_rows * local_cols * sizeof(float);
          }
          
          free(block_buffer);
        }
      }
    }
  } else {
    // Receive from the root process
    MPI_Recv(local_matrix, local_rows * local_cols, MPI_FLOAT, 
             0, 0, comm_2d, MPI_STATUS_IGNORE);
    
    // Tracking for profiling
    if (counts != NULL) {
      counts->recv_count++;
      counts->recv_bytes += local_rows * local_cols * sizeof(float);
    }
  }
}

void collect_global_message_stats(TimingInfo *timing, char variant, int rank, int nprocs, MPI_Comm comm) {
    int global_send_count = 0;
    int global_recv_count = 0;
    size_t global_send_bytes = 0;
    size_t global_recv_bytes = 0;
    
    // Reduce to get total counts across all processes
    MPI_Reduce(&timing->send_count, &global_send_count, 1, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(&timing->recv_count, &global_recv_count, 1, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(&timing->send_bytes, &global_send_bytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&timing->recv_bytes, &global_recv_bytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
    
    if (rank == 0) {
        printf("\nMessage Statistics for SUMMA Stationary %c:\n", variant);
        printf("Total Send operations:      %d\n", global_send_count);
        printf("Total Receive operations:   %d\n", global_recv_count);
        printf("Total Message operations:   %d\n", global_send_count + global_recv_count);
        printf("------------------------------------\n");
        printf("Global Data Movement:\n");
        printf("Total Sent data:            %.2f MB\n", global_send_bytes / (1024.0 * 1024.0));
        printf("Total Received data:        %.2f MB\n", global_recv_bytes / (1024.0 * 1024.0));
        printf("Total Data movement:        %.2f MB\n", (global_send_bytes + global_recv_bytes) / (1024.0 * 1024.0));
        printf("------------------------------------\n");
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
    
    timing.send_count = 0;
    timing.recv_count = 0;
    timing.send_bytes = 0;
    timing.recv_bytes = 0;
    
    start_time = MPI_Wtime();
    
    double init_start = MPI_Wtime();
    
    int grid_size = (int)sqrt(nprocs);
    
    // Create 2D process grid
    int dims[2] = {grid_size, grid_size};
    int periods[2] = {0, 0};
    MPI_Comm comm_2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2d);
    
    // Get current process coordinate
    int coords[2];
    MPI_Cart_coords(comm_2d, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
    
    // Create row and column communicators
    MPI_Comm row_comm, col_comm;
    int remain_dims[2];
    
    // Row communicator: fixed row index, varying column index
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(comm_2d, remain_dims, &row_comm);
    
    // Column communicator: varying row index, fixed column index
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(comm_2d, remain_dims, &col_comm);
    
    // local block dimensions
    int local_m = m / grid_size;
    int local_k = k / grid_size;
    int local_n = n / grid_size;
    
    float *local_A = (float *)malloc(local_m * local_k * sizeof(float));
    float *local_B = (float *)malloc(local_k * local_n * sizeof(float));
    float *B_bcast = (float *)malloc(local_k * local_n * sizeof(float));
    float *local_C = (float *)calloc(local_m * local_n, sizeof(float));
    
    // Generate matrices only on root process
    float *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = generate_matrix_A(m, k, rank);
        B = generate_matrix_B(k, n, rank);
        C = (float *)calloc(m * n, sizeof(float));
    }
    
    // Create MPI datatypes for matrix blocks
    MPI_Datatype block_type_A, block_type_B;
    
    // Create a datatype for block A
    MPI_Type_vector(local_m, local_k, k, MPI_FLOAT, &block_type_A);
    MPI_Type_commit(&block_type_A);
    
    // Create a datatype for block B
    MPI_Type_vector(local_k, local_n, n, MPI_FLOAT, &block_type_B);
    MPI_Type_commit(&block_type_B);
    
    timing.init_time = MPI_Wtime() - init_start;
    
    double dist_start = MPI_Wtime();
    
    DistributionCounts dist_counts = {0, 0, 0, 0};
    
    // Distribute initial matrix blocks
    distribute_matrix_blocks(A, local_A, m, k, grid_size, comm_2d, block_type_A, &dist_counts);
    distribute_matrix_blocks(B, local_B, k, n, grid_size, comm_2d, block_type_B, &dist_counts);
    
    timing.send_count += dist_counts.send_count;
    timing.recv_count += dist_counts.recv_count;
    timing.send_bytes += dist_counts.send_bytes;
    timing.recv_bytes += dist_counts.recv_bytes;
    
    timing.dist_time = MPI_Wtime() - dist_start;
    
    double comp_start = MPI_Wtime();
    
    memset(local_C, 0, local_m * local_n * sizeof(float));
    
    // SUMMA stationary A computation
    for (int k = 0; k < grid_size; k++) {
        // If source process (k,my_col), prepare the B block for broadcast
        if (my_row == k) {
            memcpy(B_bcast, local_B, local_k * local_n * sizeof(float));
        }
        
        // Broadcast B_kj from process(k,my_col) to all procs in my_col
        MPI_Bcast(B_bcast, local_k * local_n, MPI_FLOAT, k, col_comm);
        
        // Track messages from broadcast (one process sends to all)
        int col_size;
        MPI_Comm_size(col_comm, &col_size);
        if (my_row == k) {
            timing.send_count += (col_size - 1);
            timing.send_bytes += (col_size - 1) * local_k * local_n * sizeof(float);
        } else {
            timing.recv_count++;
            timing.recv_bytes += local_k * local_n * sizeof(float);
        }
        
        // temporary buffer for the matmul result
        float *temp_result = (float *)calloc(local_m * local_n, sizeof(float));
        
        matmul(local_A, B_bcast, temp_result, local_m, local_n, local_k);
        
        // Accumulate the result
        for (int i = 0; i < local_m * local_n; i++) {
            local_C[i] += temp_result[i];
        }
        free(temp_result);
    }
    
    // Synchronize to ensure all computation is finished
    MPI_Barrier(comm_2d);
    
    timing.comp_time = MPI_Wtime() - comp_start;
    
    double gather_start = MPI_Wtime();
    
    // Gather results from all procs to construct the final C matrix
    if (rank == 0) {
        // local result copy to global C
        for (int i = 0; i < local_m; i++) {
            for (int j = 0; j < local_n; j++) {
                C[i * n + j] = local_C[i * local_n + j];
            }
        }
        
        // results from other processes
        for (int r = 1; r < nprocs; r++) {
            int r_coords[2];
            MPI_Cart_coords(comm_2d, r, 2, r_coords);
            
            float *recv_C = (float *)malloc(local_m * local_n * sizeof(float));
            MPI_Recv(recv_C, local_m * local_n, MPI_FLOAT, r, 0, comm_2d, MPI_STATUS_IGNORE);
            timing.recv_count++;
            timing.recv_bytes += local_m * local_n * sizeof(float);
            
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
        // Send local results to root
        MPI_Send(local_C, local_m * local_n, MPI_FLOAT, 0, 0, comm_2d);
        timing.send_count++;
        timing.send_bytes += local_m * local_n * sizeof(float);
    }
    
    timing.gather_time = MPI_Wtime() - gather_start;
    timing.total_time = MPI_Wtime() - start_time;
    print_timing_info(&timing, 'A', rank);
    
    // Gather and print global message statistics
    collect_global_message_stats(&timing, 'A', rank, nprocs, MPI_COMM_WORLD);
    
    // Clean up
    free(local_A);
    free(local_B);
    free(B_bcast);
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
    
    timing.send_count = 0;
    timing.recv_count = 0;
    timing.send_bytes = 0;
    timing.recv_bytes = 0;
    
    start_time = MPI_Wtime();
    double init_start = MPI_Wtime();
    
    int grid_size = (int)sqrt(nprocs);
    
    // Create 2D process grid
    int dims[2] = {grid_size, grid_size};
    int periods[2] = {0, 0};
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
    
    // Calculate local matrix dims
    int local_m = m / grid_size;
    int local_n = n / grid_size;
    int local_k = k / grid_size;
    
    // Generate matrices only on root
    float *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = generate_matrix_A(m, k, rank);
        B = generate_matrix_B(k, n, rank);
        C = (float *)calloc(m * n, sizeof(float));
    }

    float *local_A = (float *)malloc(local_m * local_k * sizeof(float));
    float *local_B = (float *)malloc(local_k * local_n * sizeof(float));
    float *local_C = (float *)calloc(local_m * local_n, sizeof(float));
    float *temp_A = (float *)malloc(local_m * local_k * sizeof(float));
    float *temp_B = (float *)malloc(local_k * local_n * sizeof(float));
    float *temp_C = (float *)calloc(local_m * local_n, sizeof(float));
    
    // Create MPI datatypes
    MPI_Datatype block_type_A, block_type_B;
    MPI_Type_vector(local_m, local_k, k, MPI_FLOAT, &block_type_A);
    MPI_Type_commit(&block_type_A);
    
    MPI_Type_vector(local_k, local_n, n, MPI_FLOAT, &block_type_B);
    MPI_Type_commit(&block_type_B);
    
    timing.init_time = MPI_Wtime() - init_start;
    double dist_start = MPI_Wtime();
    
    DistributionCounts dist_counts = {0, 0, 0, 0};
    
    // Distribute initial matrix blocks
    distribute_matrix_blocks(A, local_A, m, k, grid_size, comm_2d, block_type_A, &dist_counts);
    distribute_matrix_blocks(B, local_B, k, n, grid_size, comm_2d, block_type_B, &dist_counts);
    
    // track message counts
    timing.send_count += dist_counts.send_count;
    timing.recv_count += dist_counts.recv_count;
    timing.send_bytes += dist_counts.send_bytes;
    timing.recv_bytes += dist_counts.recv_bytes;
    
    timing.dist_time = MPI_Wtime() - dist_start;
    double comp_start = MPI_Wtime();
    
    memset(local_C, 0, local_m * local_n * sizeof(float));
    
    // SUMMA computation with C stationary
    for (int l = 0; l < grid_size; l++) {
        // Broadcast A blocks along rows - iteration l, we need the A block from process (my_row, l)
        if (my_col == l) {
            memcpy(temp_A, local_A, local_m * local_k * sizeof(float));
        }
        
        // Broadcast A block from process (my_row, l) to all processes in the same row
        MPI_Bcast(temp_A, local_m * local_k, MPI_FLOAT, l, row_comm);
        
        // tracking message and data movement
        int row_size;
        MPI_Comm_size(row_comm, &row_size);
        if (my_col == l) {
            timing.send_count += (row_size - 1);
            timing.send_bytes += (row_size - 1) * local_m * local_k * sizeof(float);
        } else {
            timing.recv_count++;
            timing.recv_bytes += local_m * local_k * sizeof(float);
        }
        
        // Broadcast B blocks along col each iteration l, B block from process (l, my_col)
        if (my_row == l) {
            memcpy(temp_B, local_B, local_k * local_n * sizeof(float));
        }
        
        // Broadcast B block from process (l, my_col) to all procs in the same col
        MPI_Bcast(temp_B, local_k * local_n, MPI_FLOAT, l, col_comm);
        
        int col_size;
        MPI_Comm_size(col_comm, &col_size);
        if (my_row == l) {
            timing.send_count += (col_size - 1);
            timing.send_bytes += (col_size - 1) * local_k * local_n * sizeof(float);
        } else {
            timing.recv_count++;
            timing.recv_bytes += local_k * local_n * sizeof(float);
        }
        
        // local matmul
        memset(temp_C, 0, local_m * local_n * sizeof(float));
        matmul(temp_A, temp_B, temp_C, local_m, local_n, local_k);
        
        // Accumulate the result
        for (int i = 0; i < local_m * local_n; i++) {
            local_C[i] += temp_C[i];
        }
        MPI_Barrier(comm_2d);
    }
    
    timing.comp_time = MPI_Wtime() - comp_start;
    double gather_start = MPI_Wtime();
    
    // Gather results
    if (rank == 0) {
        // local result copy to global C
        for (int i = 0; i < local_m; i++) {
            for (int j = 0; j < local_n; j++) {
                C[i * n + j] = local_C[i * local_n + j];
            }
        }
        
        // results from other processes
        for (int r = 1; r < nprocs; r++) {
            int r_coords[2];
            MPI_Cart_coords(comm_2d, r, 2, r_coords);
            float *recv_C = (float *)malloc(local_m * local_n * sizeof(float));
            
            MPI_Recv(recv_C, local_m * local_n, MPI_FLOAT, r, 0, comm_2d, MPI_STATUS_IGNORE);
            timing.recv_count++;
            timing.recv_bytes += local_m * local_n * sizeof(float);
            
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
        timing.send_count++;
        timing.send_bytes += local_m * local_n * sizeof(float);
    }
    
    timing.gather_time = MPI_Wtime() - gather_start;
    timing.total_time = MPI_Wtime() - start_time;
    print_timing_info(&timing, 'C', rank);
    collect_global_message_stats(&timing, 'C', rank, nprocs, MPI_COMM_WORLD);
    
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
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
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