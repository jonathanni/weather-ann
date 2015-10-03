#ifndef _ENGINE_GUARD
#define _ENGINE_GUARD

#include <fann.h>
#include "bool.h"

#ifndef wrap
#define wrap(num, low, high) ((num) == (low - 1) ? (high - 1) : ((num) == (high) ? (low) : (num)))
#endif

#ifndef randrange
#define randrange(a, b) ((rand() % ((b) - (a))) + (a))
#endif

void train(char *);
void forecast(char *, int);

void openNetwork(void);
void closeNetwork(void);

void setNetworkOptions(void);

void readNetwork(void);
void writeNetwork(void);

void openData(char *);
void readData(fann_type *, int, int, int);
void closeData(void);

void unpackSingleLevel(int, int, int, int, int, fann_type *, bool, int);
void unpackLevel(int, int, int, int, int, int, fann_type *, int);
void unpackLevels(int, int, int, int, fann_type *, int);

void unpackShortInto(short *, fann_type *, int, float, float, float, float);
void unpackDoubleInto(double *, double *, fann_type *, int, float, float, float, float);

void openTimes(void);
void closeTimes(void);

void openNetCDF(char *, int, int);
void writeNetCDF(fann_type *, int);
void closeNetCDF(void);

void safeFree(void *);

#endif
