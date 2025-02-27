#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include "spmm.h"
#include "utils.h"

void print_usage() {
    printf("Usage: ./spmm [options]\n");
    printf("Options:\n");
    printf("For random matrix generation:\n");
    printf("  -r, --rows INT       Number of rows\n");
    printf("  -c, --cols INT       Number of columns\n");
    printf("  -d, --density FLOAT  Matrix density (0-1)\n");
    printf("\nOr for matrix market input:\n");
    printf("  -f, --file STRING    Input matrix market file\n");
    printf("\nRequired for both:\n");
    printf("  -b, --bcols INT      Number of columns in B\n");
    printf("\nOptional:\n");
    printf("  -v, --verbose        Print detailed output\n");
    printf("  -m, --metrics        Print performance metrics\n");
    printf("  -h, --help           Print this help\n");
}

int main(int argc, char *argv[]) {
    int rows = 0, cols = 0, B_cols = 0;
    float density = 0.0;
    int verbose = 0, metrics = 0;
    char *mtx_file = NULL;
    clock_t start, end;
    double cpu_time_used;

    struct option long_options[] = {
        {"rows",    required_argument, 0, 'r'},
        {"cols",    required_argument, 0, 'c'},
        {"density", required_argument, 0, 'd'},
        {"bcols",   required_argument, 0, 'b'},
        {"file",    required_argument, 0, 'f'},
        {"verbose", no_argument,       0, 'v'},
        {"metrics", no_argument,       0, 'm'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "r:c:d:b:f:vmh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'r': rows = atoi(optarg); break;
            case 'c': cols = atoi(optarg); break;
            case 'd': density = atof(optarg); break;
            case 'b': B_cols = atoi(optarg); break;
            case 'f': mtx_file = optarg; break;
            case 'v': verbose = 1; break;
            case 'm': metrics = 1; break;
            case 'h': print_usage(); return 0;
            default: print_usage(); return 1;
        }
    }

    CSRMatrix *A;
    int nnz;

    if (mtx_file) {
        // Read from matrix market file
        A = read_matrix_market_to_csr(mtx_file);
        if (!A) {
            printf("Error reading matrix market file\n");
            return 1;
        }
        rows = A->rows;
        cols = A->cols;
        nnz = A->nnz;
    } else {
        // Generate random matrix
        if (rows <= 0 || cols <= 0 || density <= 0 || density > 1) {
            printf("Invalid or missing parameters\n");
            print_usage();
            return 1;
        }
        nnz = (int)(rows * cols * density);
        if (nnz < rows) nnz = rows;
        A = create_csr_matrix(rows, cols, nnz);
        generate_random_csr(A, density);
    }

    if (B_cols <= 0) {
        printf("Invalid or missing B_cols parameter\n");
        print_usage();
        return 1;
    }
    
    float *B = (float *)malloc(cols * B_cols * sizeof(float));
    float *C = (float *)calloc(rows * B_cols, sizeof(float));
    
    for (int i = 0; i < cols * B_cols; i++) {
        B[i] = (float)(rand() % 10);
    }

    printf("\nPerforming SpMM...\n");
    
    start = clock();
    csr_spmm(A, B, C, B_cols);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    if (verbose) {
        printf("\nSparse Matrix A:\n");
        print_csr_matrix(A);
        printf("\nResult Matrix C:\n");
        print_matrix(C, rows, B_cols);
    }

    if (metrics) {
        printf("\nPerformance Metrics:\n");
        printf("Matrix dimensions: %d x %d\n", rows, cols);
        printf("Non-zeros: %d (density: %.2f%%)\n", nnz, density * 100);
        printf("B columns: %d\n", B_cols);
        printf("Execution time: %.6f seconds\n", cpu_time_used);
        // Each non-zero element requires 1 multiplication and 1 addition
        long long total_ops = (long long)nnz * B_cols * 2;
        printf("Operations performed: %lld\n", total_ops);
        printf("GFLOPS: %.2f\n", total_ops / (cpu_time_used * 1e9));
        printf("Memory used: %.2f MB\n", 
            (nnz * (sizeof(float) + sizeof(int)) + // CSR format
             (rows + 1) * sizeof(int) +            // row pointers
             cols * B_cols * sizeof(float) +       // Matrix B
             rows * B_cols * sizeof(float)         // Matrix C
            ) / (1024.0 * 1024.0));
    }

    free(B);
    free(C);
    free_csr_matrix(A);
    return 0;
}