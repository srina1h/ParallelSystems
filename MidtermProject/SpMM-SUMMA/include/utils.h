#ifndef UTILS_H
#define UTILS_H

#include "csr_matrix.h"

void generate_random_csr(CSRMatrix *A, float density);
void print_matrix(float *mat, int rows, int cols);
void print_performance_metrics(float time_taken, int flops);
CSRMatrix* read_matrix_market_to_csr(const char *filename);

#endif