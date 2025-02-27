CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall -I./include
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

TARGET = spmm

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean directories