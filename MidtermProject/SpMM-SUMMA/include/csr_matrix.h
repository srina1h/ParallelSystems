#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

typedef struct {
    float *values;
    int *col_indices;
    int *row_ptr;
    int rows;
    int cols;
    int nnz;
} CSRMatrix;

CSRMatrix* create_csr_matrix(int rows, int cols, int nnz);
void free_csr_matrix(CSRMatrix *mat);
void print_csr_matrix(CSRMatrix *mat);

#endif