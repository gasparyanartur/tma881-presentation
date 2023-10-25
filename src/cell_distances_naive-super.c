#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <omp.h>

#define N_BLOCKS_PER_CHUNK 1
#define N_CELLS_PER_BLOCK 1
#define N_COORDS_PER_CELL 3 
#define COORD_STR_LEN 8

#define INPUT_PATH_EX_SMALL "data/example-input.txt"
#define INPUT_PATH_EX_LARGE "data/large-input.txt"
#define TMP_INPUT "data/tmp-input.txt"
#define INPUT_PATH_1e4 "/home/hpc2023/cell_distances/test_data/cells_1e4"
#define INPUT_PATH_1e5 "/home/hpc2023/cell_distances/test_data/cells_1e5"

// points in [-10, 10] -> Max dist = Sqrt(20^2 + 20^2 + 20^2) = 34.64...
// converting to ints, we get 34.64 * 100 = 3464 possible distance values.
#define N_DISTS 3465

#define DTYPE int
#define S0 10000
#define S1 1000
#define S2 100
#define S3 10
#define S4 1
#define DIST_TO_INDEX 0.1f


static inline void compute_dist_between_chunks(DTYPE *chunk1, DTYPE *chunk2, uint64_t *dist_counts, size_t csz1, size_t csz2);
static inline void compute_dist_between_self_chunk(DTYPE *chunk, uint64_t *dist_counts, size_t csz);
static inline void compute_dists_between_blocks(DTYPE *block1, DTYPE *block2, uint64_t *dist_counts, size_t b1sz, size_t b2sz);
static inline uint16_t compute_dist_between_cells(DTYPE x1, DTYPE y1, DTYPE z1, DTYPE x2, DTYPE y2, DTYPE z2);
static inline void compute_dists_between_self(DTYPE *block, uint64_t *dist_counts, size_t bsz);
static inline DTYPE sign_char_to_num(char c);
static inline void read_file(FILE *stream, DTYPE *dst_arr, size_t start_cell, size_t n_cells);
static inline void print_file_chunk(char *file_str, size_t start_cell, size_t n_cells);


const char *PATHS[5] = {
    INPUT_PATH_EX_SMALL,
    INPUT_PATH_EX_LARGE,
    TMP_INPUT,
    INPUT_PATH_1e4,
    INPUT_PATH_1e5
};


int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc == 1){
        printf("Missing number of threads!\n");
        exit(-1);
    }
    const uint8_t n_threads = atoi(argv[1]+2);
    const size_t file_type = (argc > 2) ? atoi(argv[2]+2) : 0;
    const char *file_path = PATHS[file_type];

    omp_set_num_threads(n_threads);

    FILE *stream = fopen(file_path, "rb");
    if (stream == NULL) {
        printf("File error\n");
        exit(-1);
    }
    
    // Find num lines in file
    // We navigate to the end of the file and get the current file position with ftell
    // We then add one (with long type since ftell is long) just in case there is no newline at end
    // We then divide by three because there are three numbers in each line
    // Finally we divide by 8 because each number is 8 chars long (including blank space and new line)
    fseek(stream, 0L, SEEK_END);
    size_t n_cells = (size_t)((ftell(stream)+1L) / (N_COORDS_PER_CELL * COORD_STR_LEN));
    fseek(stream, 0L, SEEK_SET);

    size_t cell_size = N_COORDS_PER_CELL;

    uint64_t dist_counts[N_DISTS] = {0};

    
    for (size_t i = 0; i < n_cells; ++i) {
        DTYPE cell1[3];
        read_file(stream, cell1, i, 1);
        
        for (size_t j = i+1; j < n_cells; ++j) {
            DTYPE cell2[3];
            read_file(stream, cell2, j, 1);

            const uint16_t dist = compute_dist_between_cells(cell1[0], cell1[1], cell1[2], cell2[0], cell2[1], cell2[2]);
            dist_counts[dist]++;
        }
    }

    for (uint16_t index = 0; index < N_DISTS; ++index) {
        const float dist = index * 0.01f;
        const uint64_t count = dist_counts[index];

        if (count) {
            printf("%05.2f %lu\n", dist, count);
        }
    }
    fclose(stream);
}


static inline uint16_t compute_dist_between_cells(DTYPE x1, DTYPE y1, DTYPE z1, DTYPE x2, DTYPE y2, DTYPE z2) {
    DTYPE dx = x2 - x1;
    DTYPE dy = y2 - y1;
    DTYPE dz = z2 - z1;
    return (uint16_t)(sqrtf((float)(dx*dx + dy*dy + dz*dz)) * DIST_TO_INDEX);
}


static inline DTYPE sign_char_to_num(char c) {
    /* Return -1 for '-' and 1 for '+'. 
       The ASCII for '+' is 43, ',' is 44, and '-' is 45.
       Thus, we take 44 - c, which is 1 for '+' and -1 for '-'.
    */
    return ',' - c;
}


static inline void print_file_chunk(char *file_str, size_t start_cell, size_t n_cells) {
    for (size_t i = 0; i < n_cells; ++i) {
        printf("i: %lu\tstr: ", n_cells+i);

        for (int j = 0; j < 3*N_CELLS_PER_BLOCK; ++j) 
            printf("%c", file_str[i*3*N_CELLS_PER_BLOCK+j]);
    } 
    printf("\n");
}


static inline void read_file(FILE *stream, DTYPE *dst_arr, size_t start_cell, size_t n_cells) {
    // end line not inclusive

    const size_t n_chars = n_cells * N_COORDS_PER_CELL * COORD_STR_LEN;

    char file_str[n_chars];
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
}