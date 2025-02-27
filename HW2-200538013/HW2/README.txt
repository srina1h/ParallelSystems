Running:

To each spmv folder:
-  Add the example_matrices folder with the required test matrices
-  Add the include folder, config.h, mmio.h

make clean
make

Explanation:

All parallelization strategies show similar trends across different matrices (i.e. each technique
works simialr relative to the other techniques across every matrix) with one exception - hybrid n4 N8
seems to take longer than n2 n8 in every matrix except for bfly. But the variance seems small making it
negligeble

In general, omp seems to perform the fastest in all matrices. This could be due to the cost of splitting into
rows and sending over the network being higher than the cost to perform multi-threading within same core using
OpenMP.

By this logic, the expectation would be that the hybrid would perfrom faster than the MPI version. But this does
not seem to be the case. The hybrid versions all perform poorly compared to the pure MPI version. This could be due
to additional overhead of splitting across rows and then further parallelization using multithreading across
different nodes.

Across the hybrid versions - n8 N8 seems to be the fastest, followed by n2 N8 and then n4 N8. The expectation would
be that the n4 n8 would be faster, but the cost of splitting across 4 nodes - 2 cores per node seems to outweigh
the possible benefit.