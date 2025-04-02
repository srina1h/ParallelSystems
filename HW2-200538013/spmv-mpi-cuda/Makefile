# CC=gcc
CC=nvcc
#FLAG=-g -Wall
FLAG=-O3 -std=c++11 -I./include/ -diag-suppress 177,2464
LDFLAG=-O3

OBJS=spmv-mpi-cuda.cu mmio.c

.c.o:
	${CC} -o $@ -c ${FLAG} $<

spmv: ${OBJS}
# ${CC} -lm ${LDFLAG} -o $@ $^
	${CC} ${FLAG} -x c mmio.c -x cu spmv-mpi-cuda.cu -ccbin mpicxx -Xcompiler "-DMPICH_SKIP_MPICXX" -o $@ -lmpi

.PHONY:clean
clean: 
	find ./ -name "*.o" -delete
	rm spmv

