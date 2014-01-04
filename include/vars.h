#ifndef _VARS_GUARD
#define _VARS_GUARD

#define TEMP 0
#define HEGT 1
#define RELH 2
#define UWND 3
#define VWND 4
#define SSTP 5
#define PREW 6

#define LEARNING_RATE 0.025f

#define SIG_SCALE 6.0f

#define WIDTH 144
#define HEIGHT 73

#define HIDDEN_LAYERS 7

#define SAMPLE_SIZE 4

#define GRID_SIZE (WIDTH * HEIGHT)
#define NODE_SIZE ((HIDDEN_LAYERS + 1) * GRID_SIZE + SAMPLE_SIZE * GRID_SIZE)

#define X 77.5f
#define Y 37.5f
#define R 1

#define WEIGHT_SURF_NUM_1 (0)
#define WEIGHT_SURF_NUM_2 (9 * 7)
#define WEIGHT_SURF_NUM_3 (9 * 5)

#define WEIGHT_CSUR_NUM_1 (9 * 7)
#define WEIGHT_CSUR_NUM_2 (9 * 5)
#define WEIGHT_CSUR_NUM_3 (9 * 5)

#define WEIGHT_BODY_NUM_1 (9 * 5)
#define WEIGHT_BODY_NUM_2 (9 * 5)
#define WEIGHT_BODY_NUM_3 (9 * 5)

#define WEIGHT_TOP_NUM_1 (9 * 5)
#define WEIGHT_TOP_NUM_2 (9 * 5)
#define WEIGHT_TOP_NUM_3 (0)

#define WEIGHT_SURF_SIZE (GRID_SIZE * (9 * 7 + 9 * 5))
#define WEIGHT_CSUR_SIZE (GRID_SIZE * (9 * 7 + 9 * 5 * 2))
#define WEIGHT_BODY_SIZE (GRID_SIZE * (9 * 5 * 3))
#define WEIGHT_TOP_SIZE (GRID_SIZE * (9 * 5 * 2))

#define SST_TIME_OFFSET_1 17363592
#define SST_TIME_OFFSET_2 17435232

#define NODE_OFFSET_BLOCK (HIDDEN_LAYERS + 2)
#define WEIGHT_OFFSET_BLOCK (HIDDEN_LAYERS + 1)

#endif
