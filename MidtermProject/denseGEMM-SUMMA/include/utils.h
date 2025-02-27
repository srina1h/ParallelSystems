// utils.h
#ifndef UTILS_H
#define UTILS_H

void matmul(double *A, double *B, double *C, int m, int n, int k);
void verify_result(double *C_global, double *A, double *B, int m, int n, int k);
double *generate_matrix_A(int rows, int cols, int rank);
double *generate_matrix_B(int rows, int cols, int rank);

#endif