#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "constants.h"
#include "fileread.h"


static inline DTYPE sign_char_to_num(char c) {
    /* Return -1 for '-' and 1 for '+'. 
       The ASCII for '+' is 43, ',' is 44, and '-' is 45.
       Thus, we take 44 - c, which is 1 for '+' and -1 for '-'.
    */
    return ',' - c;
}

void print_file_chunk(char *file_str, size_t start_cell, size_t n_cells) {
    for (size_t i = 0; i < n_cells; ++i) {
        printf("i: %lu\tstr: ", n_cells+i);

        for (int j = 0; j < 3*N_CELLS_PER_BLOCK; ++j) 
            printf("%c", file_str[i*3*N_CELLS_PER_BLOCK+j]);
    } 
    printf("\n");
}


void read_file(FILE *stream, DTYPE *dst_arr, size_t start_cell, size_t n_cells) {
    // end line not inclusive

    const size_t n_chars = n_cells * N_COORDS_PER_CELL * COORD_STR_LEN;

    char *file_str = malloc(sizeof(char) * n_chars);
    fseek(stream, start_cell * N_COORDS_PER_CELL * COORD_STR_LEN * (size_t)sizeof(char), SEEK_SET);
    fread(file_str, sizeof(char), n_chars, stream);

    /* Looking at lines
        
            +01.330 -09.035 +03.489
            -03.718 +02.517 -05.995
            +09.568 -03.464 +02.645
            
        We can make the following observations:
        
        - There are three numbers per line
        - There are 8 characters for each number
        - There are 24 characters in each line

        Thus, we can directly compute the number from the strings.

            c | + 0 1 . 3 3 0
            i | 0 1 2 3 4 5 6

        Assuming we use floats, we get

            sign(w[0]) * (w[1] * 10 + w[2] * 1 + w[4] * 0.1 + w[5] * 0.01 + w[6] * 0.001)

        If we decide to go for unsigned integers instead, 
        this needs to be shifted by 10 and multiplied by 1000 to get a range [0, 20000].
    */
    const size_t n_coords = n_cells * N_COORDS_PER_CELL;
    for (size_t i = 0; i < n_coords; ++i) {
        size_t base = i * COORD_STR_LEN;
        dst_arr[i] =  (
            S0 * (file_str[base + 1] - '0') +
            S1 * (file_str[base + 2] - '0') +
            S2 * (file_str[base + 4] - '0') +
            S3 * (file_str[base + 5] - '0') + 
            S4 * (file_str[base + 6] - '0')
        ) * sign_char_to_num(file_str[base + 0]);
    }

    free(file_str);
}