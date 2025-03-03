CC = mpicc
CFLAGS = -Wall -O3
INCLUDES = -I./include
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = .

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET = summa

all: directories $(TARGET)

directories:
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/*.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

run:
	mpirun -np 4 ./summa -m 16 -n 16 -k 16 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 16 -n 16 -k 16 -b 16 -s b -v -p

runsmall1:
	mpirun -np 4 ./summa -m 4096 -n 4096 -k 4096 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 4096 -n 4096 -k 4096 -b 16 -s b -v -p

runsmall2:
	mpirun -np 4 ./summa -m 8192 -n 8192 -k 8192 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 8192 -n 8192 -k 8192 -b 16 -s b -v -p

runmedium1:
	mpirun -np 4 ./summa -m 16384 -n 16384 -k 16384 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 16384 -n 16384 -k 16384 -b 16 -s b -v -p

runmedium2:
	mpirun -np 4 ./summa -m 32768 -n 32768 -k 32768 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 32768 -n 32768 -k 32768 -b 16 -s b -v -p

runlarge:
	mpirun -np 4 ./summa -m 65536 -n 65536 -k 65536 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 65536 -n 65536 -k 65536 -b 16 -s b -v -p

runrectsmall1:
	mpirun -np 4 ./summa -m 4096 -n 4096 -k 128 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 4096 -n 4096 -k 128 -b 16 -s b -v -p

runrectsmall2:
	mpirun -np 4 ./summa -m 8192 -n 8192 -k 128 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 8192 -n 8192 -k 128 -b 16 -s b -v -p

runrectmedium1:
	mpirun -np 4 ./summa -m 16384 -n 16384 -k 128 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 16384 -n 16384 -k 128 -b 16 -s b -v -p

runrectmedium2:
	mpirun -np 4 ./summa -m 32768 -n 32768 -k 128 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 32768 -n 32768 -k 128 -b 16 -s b -v -p

runrectlarge:
	mpirun -np 4 ./summa -m 65536 -n 65536 -k 128 -b 16 -s a -v -p
	mpirun -np 4 ./summa -m 65536 -n 65536 -k 128 -b 16 -s b -v -p

.PHONY: all clean run directories