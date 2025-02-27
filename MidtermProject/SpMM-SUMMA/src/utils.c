#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "utils.h"


#define MAX_LINE_LENGTH 1024

void generate_random_csr(CSRMatrix *A, float density) {
    srand(time(NULL));
    // int total_elements = A->rows * A->cols;
    
    A->row_ptr[0] = 0;
    int current_nnz = 0;
    
    for (int i = 0; i < A->rows && current_nnz < A->nnz; i++) {
        int elements_this_row = rand() % 3 + 1;
        if (current_nnz + elements_this_row > A->nnz) {
            elements_this_row = A->nnz - current_nnz;
        }
        
        for (int j = 0; j < elements_this_row; j++) {
            A->values[current_nnz] = (float)(rand() % 10);
            A->col_indices[current_nnz] = rand() % A->cols;
            current_nnz++;
        }
        A->row_ptr[i + 1] = current_nnz;
    }
    
    for (int i = A->rows; i >= 0 && A->row_ptr[i] == 0; i--) {
        A->row_ptr[i] = current_nnz;
    }
}

void print_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

void print_performance_metrics(float time_taken, int flops) {
    printf("Time taken: %f seconds\n", time_taken);
    printf("FLOPS: %d\n", flops);
}


typedef struct {
    int row;
    int col;
    float val;
} MatrixEntry;

int compare_entries(const void *a, const void *b) {
    MatrixEntry *ea = (MatrixEntry *)a;
    MatrixEntry *eb = (MatrixEntry *)b;
    if (ea->row != eb->row)
        return ea->row - eb->row;
    return ea->col - eb->col;
}

CSRMatrix* read_matrix_market_to_csr(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Cannot open file %s\n", filename);
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    int rows, cols, nnz;
    int is_pattern = 0;
    int is_symmetric = 0;

    // Read header
    if (fgets(line, MAX_LINE_LENGTH, f) == NULL) {
        fclose(f);
        return NULL;
    }
    
    // Check matrix market format
    if (strstr(line, "pattern")) is_pattern = 1;
    if (strstr(line, "symmetric")) is_symmetric = 1;

    // Skip comments
    do {
        if (fgets(line, MAX_LINE_LENGTH, f) == NULL) {
            fclose(f);
            return NULL;
        }
    } while (line[0] == '%');

    // Read dimensions
    if (sscanf(line, "%d %d %d", &rows, &cols, &nnz) != 3) {
        fclose(f);
        return NULL;
    }

    // Allocate space for entries
    int max_entries = is_symmetric ? nnz * 2 : nnz;
    MatrixEntry *entries = (MatrixEntry *)malloc(max_entries * sizeof(MatrixEntry));
    int entry_count = 0;

    // Read entries
    for (int i = 0; i < nnz; i++) {
        if (fgets(line, MAX_LINE_LENGTH, f) == NULL) break;
        
        if (is_pattern) {
            if (sscanf(line, "%d %d", &entries[entry_count].row, 
                                     &entries[entry_count].col) != 2) continue;
            entries[entry_count].val = 1.0;
        } else {
            if (sscanf(line, "%d %d %f", &entries[entry_count].row,
                                        &entries[entry_count].col,
                                        &entries[entry_count].val) != 3) continue;
        }
        
        // Convert to 0-based indexing
        entries[entry_count].row--;
        entries[entry_count].col--;
        
        entry_count++;

        // Add symmetric entry if needed
        if (is_symmetric && entries[entry_count-1].row != entries[entry_count-1].col) {
            entries[entry_count] = entries[entry_count-1];
            int temp = entries[entry_count].row;
            entries[entry_count].row = entries[entry_count].col;
            entries[entry_count].col = temp;
            entry_count++;
        }
    }

    fclose(f);

    // Sort entries by row, then column
    qsort(entries, entry_count, sizeof(MatrixEntry), compare_entries);

    // Create CSR matrix
    CSRMatrix *mat = create_csr_matrix(rows, cols, entry_count);
    
    // Fill CSR arrays
    int current_row = 0;
    mat->row_ptr[0] = 0;
    
    for (int i = 0; i < entry_count; i++) {
        while (current_row < entries[i].row) {
            current_row++;
            mat->row_ptr[current_row] = i;
        }
        mat->values[i] = entries[i].val;
        mat->col_indices[i] = entries[i].col;
    }
    
    // Fill remaining row pointers
    for (current_row++; current_row <= rows; current_row++) {
        mat->row_ptr[current_row] = entry_count;
    }

    free(entries);
    return mat;
}
