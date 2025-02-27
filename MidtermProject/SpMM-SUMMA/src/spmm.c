// src/spmm.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "spmm.h"

//Sequential SpMM
void csr_spmm(CSRMatrix *A, float *B, float *C, int B_cols) {
    memset(C, 0, A->rows * B_cols * sizeof(float));

    for (int i = 0; i < A->rows; i++) {
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
            int j = A->col_indices[k];
            float val = A->values[k];
            for (int l = 0; l < B_cols; l++) {
                C[i * B_cols + l] += val * B[j * B_cols + l];
            }
        }
    }
}

// Students implement these functions
void stationary_a_spmm_summa(
    CSRMatrix *A, float *B, float *C, int B_cols, int grid_size,
    double *comm_time, long long *total_bytes, int *num_messages) 
{
    // TODO: Implement Stationary A SpMM SUMMA
}

void stationary_b_spmm_summa(
    CSRMatrix *A, float *B, float *C, int B_cols, int grid_size,
    double *comm_time, long long *total_bytes, int *num_messages) 
{
    // TODO: Implement Stationary B SpMM SUMMA
}

void verify_spmm(CSRMatrix *A, float *B, float *C, int B_cols) {
    float *C_ref = (float*)calloc(A->rows * B_cols, sizeof(float));
    csr_spmm(A, B, C_ref, B_cols);
    
    for (int i = 0; i < A->rows * B_cols; i++) {
        if (fabs(C[i] - C_ref[i]) > 1e-10) {
            printf("Verification failed at index %d\n", i);
            free(C_ref);
            return;
        }
    }
    printf("Verification passed\n");
    free(C_ref);
}

