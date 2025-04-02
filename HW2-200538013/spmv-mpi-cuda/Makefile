# CC=gcc
CC=mpicc
#FLAG=-g -Wall
FLAG=-O3 -ccbin=nvcc -lm -std=c99 -lcudart -I./include/ -Wno-unused-result -Wno-write-strings
# FLAG=-x cu -O3 -I./include/ -I/usr/local/opt/libomp/include
LDFLAG=-O3

OBJS=spmv.o mmio.o 

.c.o:
	${CC} -o $@ -c ${FLAG} $<

spmv: ${OBJS}
# ${CC} -lm ${LDFLAG} -o $@ $^
	${CC} -lm ${LDFLAG} -o $@ $^

.PHONY:clean
clean: 
	find ./ -name "*.o" -delete
	rm spmv

