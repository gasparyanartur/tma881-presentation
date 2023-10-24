#ifndef FILEREAD_H
#define FILEREAD_H

#include <stdlib.h>
#include "constants.h"

void read_file(FILE *stream, DTYPE *dst_arr, size_t start_cell, size_t n_cells);
void print_file_chunk(char *file_str, size_t start_cell, size_t n_cells);


#endif //FILEREAD_H