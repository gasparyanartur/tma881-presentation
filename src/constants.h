#ifndef CONSTANTS_H
#define CONSTANTS_H

#define N_BLOCKS_PER_CHUNK 106
#define N_CELLS_PER_BLOCK 1024
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


// INT
#define DTYPE int
#define S0 10000
#define S1 1000
#define S2 100
#define S3 10
#define S4 1
#define DIST_TO_INDEX 0.1f

#endif //CONSTANTS_H