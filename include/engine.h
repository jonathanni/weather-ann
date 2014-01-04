#ifndef _ENGINE_GUARD
#define _ENGINE_GUARD

#include "bool.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef wrap
#define wrap(num, low, high) ((num) == (low - 1) ? (high - 1) : ((num) == (high) ? (low) : (num)))
#endif

#ifndef sigmoid
#define sigmoid(num, scale) (1.0f / (1.0f + expf(-((num) / (float) (scale)) * 6.0f)))
#endif

#ifndef randrange
#define randrange(a, b) ((rand() % ((b) - (a))) + (a))
#endif

void alloc(void);
void dealloc(void);

void train(char *);
void forecast(char *, int);

void openNetwork(void);
void closeNetwork(void);

void readNetwork(void);
void writeNetwork(void);

FILE * safeOpen(char *);

void readNetworkSegment(FILE * file, float * arr, int size);

void safeFill(float *, size_t, FILE *, bool);

void openData(char *);
void readData(int, int, float *, float *, float *, float *, float *, float *, int);
void closeData(void);

void unpackSingleLevel(int, int, int, int, float *, bool, int, int);
void unpackLevel(int, int, int, int, int, float *, int, int);
void unpackLevels(int, int, int, int, float *, float *, float *, float *, float *, float *, int);

void unpackShortInto(short *, float *, int, float, float, float, float);
void unpackDoubleInto(double *, double *, float *, int, float, float, float, float);

void openTimes(void);
void closeTimes(void);

void updateNodes(void);
// replaced with CUDA kernel
// void updateNode(int, int, int, int, float *, float *, float *, float *, int, int, int, int);

void updateErrors(void);
// replaced with CUDA kernel
/* void updateError(int, int, float *, float *, float *, float *, float *, float *, float *, float *, 
int, int, int, int, int, int, int, bool); */

void updateWeights(void);
// replaced with CUDA kernel
// void updateWeight(int, int, float *, float *, float *, float *, float *, int, int, int);

void loopNodes(void);
void writeNetCDF(char *, int, int);

void displayNetwork(int, int, int, float *);
void checkNetworkSanity(int, int, int);

float getOutputError(void);

inline void cudaSync(void);
void cudaRead(void);
void cudaWrite(void);
void cudaError(cudaError_t);

void safeFree(void *);

void _INIT(void);
void _EXIT(void);

#endif
