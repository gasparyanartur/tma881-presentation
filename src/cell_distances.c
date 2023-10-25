#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <omp.h>

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


char *HDD_PATHS[] = {
    "benchmark-data/data-1e1.txt",
    "benchmark-data/data-1e2.txt",
    "benchmark-data/data-1e3.txt",
    "benchmark-data/data-1e4.txt",
    "benchmark-data/data-1e5.txt",
};

char *SSD_PATHS[] = {
    "/run/mount/scratch/hpcuser182/presentation/benchmark-data/data-1e1.txt",
    "/run/mount/scratch/hpcuser182/presentation/benchmark-data/data-1e2.txt",
    "/run/mount/scratch/hpcuser182/presentation/benchmark-data/data-1e3.txt",
    "/run/mount/scratch/hpcuser182/presentation/benchmark-data/data-1e4.txt",
    "/run/mount/scratch/hpcuser182/presentation/benchmark-data/data-1e5.txt",
};

int n_blocks_per_chunk = 106;
int n_cells_per_block = 1024;


int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc == 1){
        printf("Missing number of threads!\n");
        exit(-1);
    }
    const uint8_t n_threads = atoi(argv[1]+2);
    
    char *file_path;
    if (argc < 3)  {
        file_path = HDD_PATHS[0];
    }
    else {
        int path_num = atoi(argv[2]+2)-1;
        char path_type = argv[2][1];
        if (!((path_num >= 0) || (path_num < 5)) || !(path_type == 's' || path_type == 'h')) {
            printf("Could not read file type, exiting.\n");
            return -1; 
        }

        if (path_type == 's')  
            file_path = SSD_PATHS[path_num];
        else if (path_type == 'h')
            file_path = HDD_PATHS[path_num];
    }

    if (argc < 4) {
        n_blocks_per_chunk = 106;
        n_cells_per_block = 1024;
    }
    else {
        int prog_type = atoi(argv[3]+2)-1;
        if (prog_type == 0) {
            n_blocks_per_chunk = 1;
            n_cells_per_block = 1;
        }
        else if (prog_type == 1) {
            n_blocks_per_chunk = 1;
            n_cells_per_block = 4096;
        }
        else if (prog_type == 2) {
            n_blocks_per_chunk = 106;
            n_cells_per_block = 1024;
        }
        else {
            printf("Could not read prog_type, exiting.\n");
            return -1;
        }
    }

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
    size_t block_size = cell_size * n_cells_per_block;
    size_t chunk_size = block_size * n_blocks_per_chunk;

    size_t block_n_cells = n_cells_per_block;
    size_t chunk_n_cells = block_n_cells * n_blocks_per_chunk;

    DTYPE *chunk1 = (DTYPE *)malloc(chunk_size*sizeof(DTYPE));
    DTYPE *chunk2 = (DTYPE *)malloc(chunk_size*sizeof(DTYPE));
    uint64_t dist_counts[N_DISTS] = {0};

    // Create chunks
    for (size_t ib = 0; ib < n_cells; ib += chunk_n_cells)  {
        size_t ie = (ib + chunk_n_cells < n_cells) ? (ib + chunk_n_cells) : (n_cells);
        read_file(stream, chunk1, ib, ie - ib);

        compute_dist_between_self_chunk(chunk1, dist_counts, (ie - ib)*cell_size);

        for (size_t jb = ib + chunk_n_cells; jb < n_cells; jb += chunk_n_cells) {
            size_t je = (jb + chunk_n_cells < n_cells) ? (jb + chunk_n_cells) : (n_cells);
            read_file(stream, chunk2, jb, je - jb);

            compute_dist_between_chunks(chunk1, chunk2, dist_counts, (ie - ib)*cell_size, (je - jb)*cell_size);
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
    free(chunk1);
    free(chunk2);
}

static inline void compute_dist_between_chunks(DTYPE *chunk1, DTYPE *chunk2, uint64_t *dist_counts, size_t csz1, size_t csz2) {
    size_t block_size = N_COORDS_PER_CELL * n_cells_per_block;
    for (size_t i = 0; i < csz1; i += N_COORDS_PER_CELL*n_cells_per_block) {
        DTYPE *block1_start = chunk1 + i;
        size_t bsz1 = block1_start + block_size < chunk1 + csz1 ? block_size : (csz1 - i);

        for (size_t j = 0; j < csz2; j += N_COORDS_PER_CELL*n_cells_per_block) {
            DTYPE *block2_start = chunk2 + j;
            size_t bsz2 = block2_start + block_size < chunk2 + csz2 ? block_size : (csz2 - j);

            compute_dists_between_blocks(block1_start, block2_start, dist_counts, bsz1, bsz2);
        }
    }
}

// csz is the number of coords in a chunk
static inline void compute_dist_between_self_chunk(DTYPE *chunk, uint64_t *dist_counts, size_t csz) {
    size_t block_size = N_COORDS_PER_CELL * n_cells_per_block;
    for (size_t i = 0; i < csz; i += block_size) {
        DTYPE *block1_start = chunk + i;
        size_t bsz1 = block1_start + block_size < chunk + csz ? block_size : (csz - i);

        compute_dists_between_self(block1_start, dist_counts, bsz1);
        
        for (size_t j = i + block_size; j < csz; j += block_size) {
            DTYPE *block2_start = chunk + j;
            size_t bsz2 = block2_start + block_size < chunk + csz ? block_size : (csz - j);
            
            compute_dists_between_blocks(block1_start, block2_start, dist_counts, bsz1, bsz2);
        }
    }
}

static inline uint16_t compute_dist_between_cells(DTYPE x1, DTYPE y1, DTYPE z1, DTYPE x2, DTYPE y2, DTYPE z2) {
    DTYPE dx = x2 - x1;
    DTYPE dy = y2 - y1;
    DTYPE dz = z2 - z1;
    return (uint16_t)(sqrtf((float)(dx*dx + dy*dy + dz*dz)) * DIST_TO_INDEX);
}

static inline void compute_dists_between_blocks(DTYPE *block1, DTYPE *block2, uint64_t *dist_counts, size_t b1sz, size_t b2sz) {
    #pragma omp parallel for reduction(+:dist_counts[:N_DISTS])
    for (size_t i = 0; i < b1sz; i += N_COORDS_PER_CELL) {
        for (size_t j = 0; j < b2sz; j += N_COORDS_PER_CELL) {
            const uint16_t dist = compute_dist_between_cells(block1[i], block1[i+1], block1[i+2], block2[j], block2[j+1], block2[j+2]);
            ++dist_counts[dist];
        }
    }
}

static inline void compute_dists_between_self(DTYPE *block, uint64_t *dist_counts, size_t bsz) {
    #pragma omp parallel for 
    for (size_t i = 0; i < bsz; i += N_COORDS_PER_CELL) {
        for (size_t j = i + N_COORDS_PER_CELL; j < bsz; j += N_COORDS_PER_CELL) {
            const uint16_t dist = compute_dist_between_cells(block[i], block[i+1], block[i+2], block[j], block[j+1], block[j+2]);
            #pragma omp atomic
            ++dist_counts[dist];
        }
    }
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

        for (int j = 0; j < 3*n_cells_per_block; ++j) 
            printf("%c", file_str[i*3*n_cells_per_block+j]);
    } 
    printf("\n");
}


static inline void read_file(FILE *stream, DTYPE *dst_arr, size_t start_cell, size_t n_cells) {
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