CC=gcc
#FLAG=-g -Wall
# FLAG=-O3 -lm -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings
FLAG=-O3 -std=c99 -I./include/ -I/usr/local/opt/libomp/include -Wno-unused-result -Wno-write-strings -fopenmp
LDFLAG=-O3

OBJS=spmv.o mmio.o 

.c.o:
	${CC} -o $@ -c ${FLAG} $<

spmv: ${OBJS}
	${CC} -lm ${LDFLAG} -o $@ $^

.PHONY:clean
clean: 
	find ./ -name "*.o" -delete
	rm spmv

