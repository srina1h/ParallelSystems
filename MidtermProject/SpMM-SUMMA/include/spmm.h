#ifndef SPMM_H
#define SPMM_H

#include "csr_matrix.h"


//Sequential SpMM
void csr_spmm(CSRMatrix *A, float *B, float *C, int B_cols);
void verify_spmm(CSRMatrix *A, float *B, float *C, int B_cols);

// TODO: Parallel SpMM SUMMA
void stationary_a_spmm_summa(
    CSRMatrix *A, 
    float *B, 
    float *C, 
    int B_cols, 
    int grid_size,
    double *comm_time, 
    long long *total_bytes, 
    int *num_messages
);

void stationary_b_spmm_summa(
    CSRMatrix *A, 
    float *B, 
    float *C, 
    int B_cols, 
    int grid_size,
    double *comm_time, 
    long long *total_bytes, 
    int *num_messages
);

#endif
