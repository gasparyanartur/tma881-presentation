#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <omp.h>

#include "constants.h"
#include "fileread.h"

static inline void compute_dist_between_chunks(DTYPE *chunk1, DTYPE *chunk2, uint64_t *dist_counts, size_t csz1, size_t csz2);
static inline void compute_dist_between_self_chunk(DTYPE *chunk, uint64_t *dist_counts, size_t csz);
static inline void compute_dists_between_blocks(DTYPE *block1, DTYPE *block2, uint64_t *dist_counts, size_t b1sz, size_t b2sz);
static inline uint16_t compute_dist_between_cells(DTYPE x1, DTYPE y1, DTYPE z1, DTYPE x2, DTYPE y2, DTYPE z2);
static inline void compute_dists_between_self(DTYPE *block, uint64_t *dist_counts, size_t bsz);


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
    size_t block_size = cell_size * N_CELLS_PER_BLOCK;
    size_t chunk_size = block_size * N_BLOCKS_PER_CHUNK;

    size_t block_n_cells = N_CELLS_PER_BLOCK;
    size_t chunk_n_cells = block_n_cells * N_BLOCKS_PER_CHUNK;

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
    size_t block_size = N_COORDS_PER_CELL * N_CELLS_PER_BLOCK;
    for (size_t i = 0; i < csz1; i += N_COORDS_PER_CELL*N_CELLS_PER_BLOCK) {
        DTYPE *block1_start = chunk1 + i;
        size_t bsz1 = block1_start + block_size < chunk1 + csz1 ? block_size : (csz1 - i);

        for (size_t j = 0; j < csz2; j += N_COORDS_PER_CELL*N_CELLS_PER_BLOCK) {
            DTYPE *block2_start = chunk2 + j;
            size_t bsz2 = block2_start + block_size < chunk2 + csz2 ? block_size : (csz2 - j);

            compute_dists_between_blocks(block1_start, block2_start, dist_counts, bsz1, bsz2);
        }
    }
}

// csz is the number of coords in a chunk
static inline void compute_dist_between_self_chunk(DTYPE *chunk, uint64_t *dist_counts, size_t csz) {
    size_t block_size = N_COORDS_PER_CELL * N_CELLS_PER_BLOCK;
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