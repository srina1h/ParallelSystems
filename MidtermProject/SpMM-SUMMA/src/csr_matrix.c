#include <stdlib.h>
#include <stdio.h>
#include "csr_matrix.h"

CSRMatrix* create_csr_matrix(int rows, int cols, int nnz) {
    CSRMatrix *mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    mat->values = (float*)malloc(nnz * sizeof(float));
    mat->col_indices = (int*)malloc(nnz * sizeof(int));
    mat->row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    mat->rows = rows;
    mat->cols = cols;
    mat->nnz = nnz;
    return mat;
}

void free_csr_matrix(CSRMatrix *mat) {
    free(mat->values);
    free(mat->col_indices);
    free(mat->row_ptr);
    free(mat);
}

void print_csr_matrix(CSRMatrix *mat) {
    printf("CSR Matrix %dx%d with %d non-zeros\n", mat->rows, mat->cols, mat->nnz);
    printf("Values: ");
    for(int i = 0; i < mat->nnz; i++) {
        printf("%f ", mat->values[i]);
    }
    printf("\n");
}