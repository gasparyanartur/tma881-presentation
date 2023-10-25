src_path = src# .c files and .h files
asm_path = asm# .s files
bin_path = bin

CC = gcc
CFLAGS = -O2 -march=native -lm -fopenmp -g
BINS = \
	$(bin_path)/cell_distances \

.PHONY: all
all: $(BINS)

$(bin_path)/cell_distances: $(src_path)/cell_distances.c
	$(CC) $(CFLAGS) $^ -o $@


.PHONY: clean
clean: 
	rm $(BINS)
