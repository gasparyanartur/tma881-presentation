src_path = src# .c files and .h files
asm_path = asm# .s files
bin_path = bin

CC = gcc
CFLAGS = -O2 -march=native -lm -fopenmp -g
BINS = \
	$(bin_path)/cell_distances-naive \
	$(bin_path)/cell_distances-improved \
	$(bin_path)/cell_distances-final \

.PHONY: all
all: $(BINS)

$(bin_path)/cell_distances-naive: $(src_path)/cell_distances_naive-super.c
	$(CC) $(CFLAGS) $^ -o $@

$(bin_path)/cell_distances-improved: $(src_path)/cell_distances_improved.c
	$(CC) $(CFLAGS) $^ -o $@

$(bin_path)/cell_distances-final: $(src_path)/cell_distances_final.c
	$(CC) $(CFLAGS) $^ -o $@


.PHONY: clean
clean: 
	rm $(BINS)
