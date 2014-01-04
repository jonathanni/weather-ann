extern "C"
{
  #include "include/engine.h"
  #include "include/vars.h"
  #include "include/bool.h"
  #include "include/logger.h"
  #include "include/error.h"

  #include <stdio.h>
  #include <stdlib.h>
  #include <time.h>
  #include <string.h>
  #include <netcdf.h>
  #include <math.h>
  #include <cuda.h>
  #include <cuda_runtime_api.h>

  // Nodes
  float * n1000 = 0;
  float *  n925 = 0;
  float *  n850 = 0;
  float *  n700 = 0;
  float *  n500 = 0;
  float *  n250 = 0;

  // GPU Nodes

  float * gn1000 = 0;
  float *  gn925 = 0;
  float *  gn850 = 0;
  float *  gn700 = 0;
  float *  gn500 = 0;
  float *  gn250 = 0;

  // Next nodes

  float * nn1000 = 0;
  float *  nn925 = 0;
  float *  nn850 = 0;
  float *  nn700 = 0;
  float *  nn500 = 0;
  float *  nn250 = 0;

  // GPU Next nodes

  float * gnn1000 = 0;
  float *  gnn925 = 0;
  float *  gnn850 = 0;
  float *  gnn700 = 0;
  float *  gnn500 = 0;
  float *  gnn250 = 0;
  
  // n250 actually represents 300 hPa because of data restrictions

  // Weights

  float * w1000 = 0;
  float *  w925 = 0;
  float *  w850 = 0;
  float *  w700 = 0;
  float *  w500 = 0;
  float *  w250 = 0;

  // GPU Weights

  float * gw1000 = 0;
  float *  gw925 = 0;
  float *  gw850 = 0;
  float *  gw700 = 0;
  float *  gw500 = 0;
  float *  gw250 = 0;

  // Errors

  float * e1000 = 0;
  float *  e925 = 0;
  float *  e850 = 0;
  float *  e700 = 0;
  float *  e500 = 0;
  float *  e250 = 0;

  // GPU Errors

  float * ge1000 = 0;
  float *  ge925 = 0;
  float *  ge850 = 0;
  float *  ge700 = 0;
  float *  ge500 = 0;
  float *  ge250 = 0;

  // Node and weight offsets
  int * NODE_OFFSETS = 0;
  int * WEIGHT_OFFSETS = 0;

  int * H_NODE_OFFSETS = 0;
  int * H_WEIGHT_OFFSETS = 0;
  
  // Files for nodes

  FILE * fn1000 = 0;
  FILE *  fn925 = 0;
  FILE *  fn850 = 0;
  FILE *  fn700 = 0;
  FILE *  fn500 = 0;
  FILE *  fn250 = 0;

  // Files for weights

  FILE * fw1000 = 0;
  FILE *  fw925 = 0;
  FILE *  fw850 = 0;
  FILE *  fw700 = 0;
  FILE *  fw500 = 0;
  FILE *  fw250 = 0;

  // NetCDF ncids for files

  int hgt_ncid = 0;
  int pw_ncid = 0;
  int rh_ncid = 0;
  int sst_ncid = 0;
  int tmp_ncid = 0;
  int uwnd_ncid = 0;
  int vwnd_ncid = 0;
  int mask_ncid = 0;

  // File is empty

  bool empty = false;

  // Time arrays

  double * atime = 0;
  double * astime = 0;

  // Size of time arrays

  size_t timelen = 0;
  size_t stimelen = 0;

  // Min and Max for variable ranges (see what I did there? var_min -> varmin?)
  static float var_min[7], var_max[7];

  __device__ float toterr = 0;

  //
  FILE * debug = 0;
  //

  void _INIT()
  {
    srand(time(NULL));

    int mul[6] = {7, 5, 5, 5, 5, 5};
    int wbases[6] = {WEIGHT_SURF_SIZE, WEIGHT_CSUR_SIZE, WEIGHT_BODY_SIZE, \
		     WEIGHT_BODY_SIZE, WEIGHT_BODY_SIZE, WEIGHT_TOP_SIZE};

    H_NODE_OFFSETS = (int *) calloc(6 * NODE_OFFSET_BLOCK, sizeof(int));
    H_WEIGHT_OFFSETS = (int *) calloc(6 * WEIGHT_OFFSET_BLOCK, sizeof(int));

    int i = 0, j = 0;
    for(i = 0; i < 6; i++)
    {
      H_NODE_OFFSETS[i * NODE_OFFSET_BLOCK] = 0;
      H_NODE_OFFSETS[i * NODE_OFFSET_BLOCK + 1] = SAMPLE_SIZE * mul[i] * GRID_SIZE;

      for(j = 2; j < NODE_OFFSET_BLOCK; j++)
        H_NODE_OFFSETS[i * NODE_OFFSET_BLOCK + j] = H_NODE_OFFSETS[i * NODE_OFFSET_BLOCK + j - 1] + mul[i] * GRID_SIZE;
    }

    for(i = 0; i < 6; i++)
    {
      H_WEIGHT_OFFSETS[i * WEIGHT_OFFSET_BLOCK] = 0;
      H_WEIGHT_OFFSETS[i * WEIGHT_OFFSET_BLOCK + 1] = SAMPLE_SIZE * mul[i] * wbases[i];

      for(j = 2; j < WEIGHT_OFFSET_BLOCK; j++)
        H_WEIGHT_OFFSETS[i * WEIGHT_OFFSET_BLOCK + j] = H_WEIGHT_OFFSETS[i * WEIGHT_OFFSET_BLOCK + j - 1] + mul[i] * wbases[i];
    }

    cudaError(cudaMalloc((void **) &NODE_OFFSETS, 6 * NODE_OFFSET_BLOCK * sizeof(int)));
    cudaError(cudaMalloc((void **) &WEIGHT_OFFSETS, 6 * WEIGHT_OFFSET_BLOCK * sizeof(int)));

    printf("%d %d %d %d %d %d %d\n", H_NODE_OFFSETS[0], H_NODE_OFFSETS[1], H_NODE_OFFSETS[2], H_NODE_OFFSETS[3], H_NODE_OFFSETS[4], H_NODE_OFFSETS[5], H_NODE_OFFSETS[6]);

    cudaError(cudaMemcpy(NODE_OFFSETS, H_NODE_OFFSETS, 6 * NODE_OFFSET_BLOCK * sizeof(int), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(WEIGHT_OFFSETS, H_WEIGHT_OFFSETS, 6 * WEIGHT_OFFSET_BLOCK * sizeof(int), cudaMemcpyHostToDevice));
  }

  void _EXIT()
  {
    safeFree(H_NODE_OFFSETS);
    safeFree(H_WEIGHT_OFFSETS);

    cudaError(cudaFree(NODE_OFFSETS));
    cudaError(cudaFree(WEIGHT_OFFSETS));
  }

  void alloc()
  {
    loginfo("Allocating memory\n");
    n1000 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE, sizeof(float));
    n925 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));
    n850 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));
    n700 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));
    n500 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));
    n250 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));

    nn1000 = (float *) calloc(7 * GRID_SIZE, sizeof(float));
    nn925 = (float *) calloc(5 * GRID_SIZE, sizeof(float));
    nn850 = (float *) calloc(5 * GRID_SIZE, sizeof(float));
    nn700 = (float *) calloc(5 * GRID_SIZE, sizeof(float));
    nn500 = (float *) calloc(5 * GRID_SIZE, sizeof(float));
    nn250 = (float *) calloc(5 * GRID_SIZE, sizeof(float));

    e1000 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE, sizeof(float));
    e925 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));
    e850 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));
    e700 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));
    e500 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));
    e250 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, sizeof(float));

    w1000 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS) * 7 * WEIGHT_SURF_SIZE, sizeof(float));
    w925 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_CSUR_SIZE, sizeof(float));
    w850 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, sizeof(float));
    w700 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, sizeof(float));
    w500 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, sizeof(float));
    w250 = (float *) calloc((SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_TOP_SIZE, sizeof(float));
  
    cudaError(cudaMalloc((void **) &gn1000, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gn925, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gn850, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gn700, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gn500, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gn250, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));

    cudaError(cudaMalloc((void **) &gnn1000, 7 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gnn925, 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gnn850, 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gnn700, 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gnn500, 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gnn250, 5 * GRID_SIZE * sizeof(float)));
  
    cudaError(cudaMalloc((void **) &ge1000, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &ge925, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &ge850, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &ge700, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &ge500, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &ge250, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float)));
    
    cudaError(cudaMalloc((void **) &gw1000, (SAMPLE_SIZE + HIDDEN_LAYERS) * 7 * WEIGHT_SURF_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gw925, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_CSUR_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gw850, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gw700, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gw500, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float)));
    cudaError(cudaMalloc((void **) &gw250, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_TOP_SIZE * sizeof(float)));
  }

  void dealloc()
  {
    loginfo("Deallocating memory\n");
    safeFree(n1000);
    safeFree(n925);
    safeFree(n850);
    safeFree(n700);
    safeFree(n500);
    safeFree(n250);

    safeFree(nn1000);
    safeFree(nn925);
    safeFree(nn850);
    safeFree(nn700);
    safeFree(nn500);
    safeFree(nn250);
    
    safeFree(e1000);
    safeFree(e925);
    safeFree(e850);
    safeFree(e700);
    safeFree(e500);
    safeFree(e250);

    safeFree(w1000);
    safeFree(w925);
    safeFree(w850);
    safeFree(w700);
    safeFree(w500);
    safeFree(w250);
  
    cudaError(cudaFree(gn1000));
    cudaError(cudaFree(gn925));
    cudaError(cudaFree(gn850));
    cudaError(cudaFree(gn700));
    cudaError(cudaFree(gn500));
    cudaError(cudaFree(gn250));
  
    cudaError(cudaFree(gnn1000));
    cudaError(cudaFree(gnn925));
    cudaError(cudaFree(gnn850));
    cudaError(cudaFree(gnn700));
    cudaError(cudaFree(gnn500));
    cudaError(cudaFree(gnn250));
 
    cudaError(cudaFree(ge1000));
    cudaError(cudaFree(ge925));
    cudaError(cudaFree(ge850));
    cudaError(cudaFree(ge700));
    cudaError(cudaFree(ge500));
    cudaError(cudaFree(ge250));

    cudaError(cudaFree(gw1000));
    cudaError(cudaFree(gw925));
    cudaError(cudaFree(gw850));
    cudaError(cudaFree(gw700));
    cudaError(cudaFree(gw500));
    cudaError(cudaFree(gw250));
  }

  void openNetwork()
  {
    fn1000 = safeOpen("../network/fn1000.dat");
    safeFill(n1000, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE, fn1000, false);
    fn925  = safeOpen("../network/fn925.dat");
    safeFill(n925, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn925, false);
    fn850  = safeOpen("../network/fn850.dat");
    safeFill(n850, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn850, false);
    fn700  = safeOpen("../network/fn700.dat");
    safeFill(n700, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn700, false);
    fn500  = safeOpen("../network/fn500.dat");
    safeFill(n500, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn500, false);
    fn250  = safeOpen("../network/fn250.dat");
    safeFill(n250, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn250, false);

    fw1000 = safeOpen("../network/fw1000.dat");
    safeFill(w1000, (SAMPLE_SIZE + HIDDEN_LAYERS) * 7 * WEIGHT_SURF_SIZE, fw1000, true);
    fw925  = safeOpen("../network/fw925.dat");
    safeFill(w925, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_CSUR_SIZE, fw925, true);
    fw850  = safeOpen("../network/fw850.dat");
    safeFill(w850, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, fw850, true);
    fw700  = safeOpen("../network/fw700.dat");
    safeFill(w700, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, fw700, true);
    fw500  = safeOpen("../network/fw500.dat");
    safeFill(w500, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, fw500, true);
    fw250  = safeOpen("../network/fw250.dat");
    safeFill(w250, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_TOP_SIZE, fw250, true);
  }

  void safeFill(float * arr, size_t size, FILE * fp, bool isWeight)
  {
    if(!empty)
      return;

    float mul = isWeight ? (((float) RAND_MAX) / 2) : ((float) RAND_MAX);
    float sub = isWeight ? (((float) RAND_MAX) / 2) : 0;
    float scale = 1.0f;

    int i = 0;
    for(i = 0; i < size; i++)
      arr[i] = scale * ((float) (rand() - sub)) / ((float) mul);

    fwrite(arr, sizeof(float), size, fp);
    fflush(fp);
  }

  FILE * safeOpen(char * name)
  {
    char inf[128];
    strcpy(inf, "Opening file: ");
    strcat(inf, name);
    strcat(inf, "\n");
    loginfo(inf);
    
    FILE * file = fopen(name, "r+");
    if(file == NULL)
    {
      loginfo("File does not exist so creating a new one.\n");
      empty = true;
      return fopen(name, "w+");
    }
    
    empty = false;
    return fopen(name, "r+");
  }

  void cudaWriteLevel(float * host, float * device, int mul, int base)
  {
    cudaError(cudaMemcpy(host, device, mul * base * sizeof(float), cudaMemcpyDeviceToHost));
  }

  void cudaRead()
  {
    loginfo("Read to CUDA from host\n");
    cudaError(cudaMemcpy(gn1000, n1000, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gn925, n925, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gn850, n850, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gn700, n700, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gn500, n500, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gn250, n250, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cudaError(cudaMemcpy(gnn1000, nn1000, 7 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gnn925, nn925, 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gnn850, nn850, 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gnn700, nn700, 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gnn500, nn500, 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gnn250, nn250, 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cudaError(cudaMemcpy(ge1000, e1000, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(ge925, e925, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(ge850, e850, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(ge700, e700, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(ge500, e500, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(ge250, e250, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cudaError(cudaMemcpy(gw1000, w1000, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 7 * WEIGHT_SURF_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gw925, w925, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_CSUR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gw850, w850, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gw700, w700, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gw500, w500, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaError(cudaMemcpy(gw250, w250, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_TOP_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  }  
  
  void cudaWrite()
  {
    loginfo("Write to host from CUDA\n");
    cudaError(cudaMemcpy(n1000, gn1000, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(n925, gn925, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(n850, gn850, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(n700, gn700, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(n500, gn500, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(n250, gn250, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cudaError(cudaMemcpy(nn1000, gnn1000, 7 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(nn925, gnn925, 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(nn850, gnn850, 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(nn700, gnn700, 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(nn500, gnn500, 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(nn250, gnn250, 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaError(cudaMemcpy(e1000, ge1000, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(e925, ge925, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(e850, ge850, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(e700, ge700, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(e500, ge500, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(e250, ge250, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cudaError(cudaMemcpy(w1000, gw1000, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 7 * WEIGHT_SURF_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(w925, gw925, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_CSUR_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(w850, gw850, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(w700, gw700, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(w500, gw500, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaError(cudaMemcpy(w250, gw250, \
              (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_TOP_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  }

  void readNetworkSegment(FILE * file, float * arr, int size)
  {
    if(fread(arr, sizeof(float), size, file) != size)
    {
      logerr("File read error: file\n"); 
      exit(EXIT_FAILURE); 
    }
  }

  void readNetwork()
  {

    loginfo("Reading network (only once)\n");

    rewind(fn1000);
    rewind(fn925);
    rewind(fn850);
    rewind(fn700);
    rewind(fn500);
    rewind(fn250);

    rewind(fw1000);
    rewind(fw925);
    rewind(fw850);
    rewind(fw700);
    rewind(fw500);
    rewind(fw250);

    loginfo("Rewind...\n");  

    readNetworkSegment(fn1000, n1000, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE);
    loginfo("Read 1000 hPa\n");
    readNetworkSegment(fn925, n925, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE);
    loginfo("Read 925 hPa\n");
    readNetworkSegment(fn850, n850, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE);
    loginfo("Read 850 hPa\n");
    readNetworkSegment(fn700, n700, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE);
    loginfo("Read 700 hPa\n");
    readNetworkSegment(fn500, n500, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE);
    loginfo("Read 500 hPa\n");
    readNetworkSegment(fn250, n250, (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE);
    loginfo("Read 250 hPa\n");
    
    readNetworkSegment(fw1000, w1000, (SAMPLE_SIZE + HIDDEN_LAYERS) * 7 * WEIGHT_SURF_SIZE);
    loginfo("Read 1000 hPa weights\n");
    readNetworkSegment(fw925, w925, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_CSUR_SIZE);
    loginfo("Read 925 hPa weights\n");
    readNetworkSegment(fw850, w850, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE);
    loginfo("Read 850 hPa weights\n");
    readNetworkSegment(fw700, w700, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE);
    loginfo("Read 700 hPa weights\n");
    readNetworkSegment(fw500, w500, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE);
    loginfo("Read 500 hPa weights\n");
    readNetworkSegment(fw250, w250, (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_TOP_SIZE);
    loginfo("Read 250 hPa weights\n");
  
  }

  void writeNetwork()
  {

    rewind(fn1000);
    rewind(fn925);
    rewind(fn850);
    rewind(fn700);
    rewind(fn500);
    rewind(fn250);

    rewind(fw1000);
    rewind(fw925);
    rewind(fw850);
    rewind(fw700);
    rewind(fw500);
    rewind(fw250);
    loginfo("Rewind...\n");
    
    fwrite(n1000, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 7 * GRID_SIZE, fn1000);
    loginfo("Write 1000 hPa\n");
    fwrite(n925, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn925);
    loginfo("Write 925 hPa\n");
    fwrite(n850, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn850);
    loginfo("Write 850 hPa\n");
    fwrite(n700, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn700);
    loginfo("Write 700 hPa\n");
    fwrite(n500, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn500);
    loginfo("Write 500 hPa\n");
    fwrite(n250, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS + 1) * 5 * GRID_SIZE, fn250);
    loginfo("Write 250 hPa\n");

    fwrite(w1000, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS) * 7 * WEIGHT_SURF_SIZE, fw1000);
    loginfo("Write 1000 hPa weights\n");
    fwrite(w925, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_CSUR_SIZE, fw925);
    loginfo("Write 925 hPa weights\n");
    fwrite(w850, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, fw850);
    loginfo("Write 850 hPa weights\n");
    fwrite(w700, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, fw700);
    loginfo("Write 700 hPa weights\n");
    fwrite(w500, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_BODY_SIZE, fw500);
    loginfo("Write 500 hPa weights\n");
    fwrite(w250, sizeof(float), (SAMPLE_SIZE + HIDDEN_LAYERS) * 5 * WEIGHT_TOP_SIZE, fw250);
    loginfo("Write 250 hPa weights\n");
  }

  void closeNetwork()
  {
    fclose(fn1000);
    fclose(fn925);
    fclose(fn850);
    fclose(fn700);
    fclose(fn500);
    fclose(fn250);

    fclose(fw1000);
    fclose(fw925);
    fclose(fw850);
    fclose(fw700);
    fclose(fw500);
    fclose(fw250);

    loginfo("Closed network\n");
  }

  void openData(char * year)
  {
    char buf[128];
    
    printf("%s ", "Heights...");
    strcpy(buf, "../data/HGT/hgt.");
    strcat(buf, year);
    strcat(buf, ".nc");
    e_netcdf(nc_open(buf, 0, &hgt_ncid));
    printf("%s\n", "OK");

    printf("%s ", "Precipitable Water...");
    strcpy(buf, "../data/PW/pr_wtr.eatm.");
    strcat(buf, year);
    strcat(buf, ".nc");
    e_netcdf(nc_open(buf, 0, &pw_ncid)); 
    printf("%s\n", "OK");

    printf("%s ", "Relative Humidity...");
    strcpy(buf, "../data/RH/rhum.");
    strcat(buf, year);
    strcat(buf, ".nc");
    e_netcdf(nc_open(buf, 0, &rh_ncid)); 
    printf("%s\n", "OK");
    
    strcpy(buf, "../data/SST/sst.");
    
    if(strtol(year, NULL, 10) < 1990)
      strcat(buf, "1");
    else
      strcat(buf, "2");

    printf("%s ", "Sea Surface Temperatures...");
    strcat(buf, ".nc");
    e_netcdf(nc_open(buf, 0, &sst_ncid)); 
    printf("%s\n", "OK");
    
    printf("%s ", "Air Temperatures...");
    strcpy(buf, "../data/TMP/air.");
    strcat(buf, year);
    strcat(buf, ".nc");
    e_netcdf(nc_open(buf, 0, &tmp_ncid)); 
    printf("%s\n", "OK");

    printf("%s ", "U Component Wind...");
    strcpy(buf, "../data/UWND/uwnd.");
    strcat(buf, year);
    strcat(buf, ".nc");
    e_netcdf(nc_open(buf, 0, &uwnd_ncid));
    printf("%s\n", "OK");

    printf("%s ", "V Component Wind...");
    strcpy(buf, "../data/VWND/vwnd.");
    strcat(buf, year);
    strcat(buf, ".nc");
    e_netcdf(nc_open(buf, 0, &vwnd_ncid));
    printf("%s\n", "OK");

    printf("%s ", "Land Sea Mask...");
    strcpy(buf, "../data/SST/lsmask.1.nc");
    e_netcdf(nc_open(buf, 0, &mask_ncid));
    printf("%s\n", "OK");

    strcpy(buf, "Opened all NetCDF files for the year ");
    strcat(buf, year);
    strcat(buf, "\n");
    loginfo(buf);
  }

  void readData(int timeind, int ssttimeind, \
                float * _n1000, float * _n925, float * _n850, float * _n700, float * _n500, float * _n250, int len)
  {
    int hgt_varid = 0;
    int pw_varid = 0;
    int rh_varid = 0;
    int sst_varid = 0;
    int tmp_varid = 0;
    int uwnd_varid = 0;
    int vwnd_varid = 0;

    e_netcdf(nc_inq_varid(hgt_ncid, "hgt", &hgt_varid));
    e_netcdf(nc_inq_varid(pw_ncid, "pr_wtr", &pw_varid));
    e_netcdf(nc_inq_varid(rh_ncid, "rhum", &rh_varid));
    e_netcdf(nc_inq_varid(sst_ncid, "sst", &sst_varid));
    e_netcdf(nc_inq_varid(tmp_ncid, "air", &tmp_varid));
    e_netcdf(nc_inq_varid(uwnd_ncid, "uwnd", &uwnd_varid));
    e_netcdf(nc_inq_varid(vwnd_ncid, "vwnd", &vwnd_varid));

    int ncids[5] = {hgt_ncid, rh_ncid, tmp_ncid, uwnd_ncid, vwnd_ncid};
    int varids[5] = {hgt_varid, rh_varid, tmp_varid, uwnd_varid, vwnd_varid};
    int sectors[5] = {HEGT, RELH, TEMP, UWND, VWND};

    int i = 0;
    for(i = 0; i < 5; i++)
      unpackLevels(timeind, ncids[i], varids[i], sectors[i], _n1000, _n925, _n850, _n700, _n500, _n250, len);
    unpackSingleLevel(timeind, pw_ncid, pw_varid, PREW, _n1000, false, 7 * GRID_SIZE, len);
    unpackSingleLevel(ssttimeind, sst_ncid, sst_varid, SSTP, _n1000, true, 7 * GRID_SIZE, len); 
  }

  void unpackSingleLevel(int timeind, int ncid, int varid, int sector, float * node, bool isConverted, int offset, int len)
  {
    // If the NetCDF file was converted using OpenGRaDS then it is of dimension 145x73, not 144x73
    // but thankfully the NetCDF nc_get_vara_type function supports selecting blocks of data more
    // gracefully. The fact that it is double * instead of short * is also noted. Also, all metadata
    // is lost for converted files so that is manually inputted (scale factor and offset).

    size_t lstart[3] = {timeind, 0, 0};
    size_t lcount[3] = {1, HEIGHT, WIDTH};

    size_t istart[2] = {0, 0};
    size_t icount[2] = {HEIGHT - 1, WIDTH}; 
   
    float scale_factor = 0;
    float add_offset = 0;

    float bounds[2];

    if(isConverted)
    {
      double * buf = (double *) calloc(GRID_SIZE, sizeof(double));
      double * mask = (double *) calloc(GRID_SIZE, sizeof(double));

      scale_factor = 0.01f;
      add_offset = 0;

      bounds[0] = -5.0f;
      bounds[1] = 40.0f;

      var_min[sector] = bounds[0];
      var_max[sector] = bounds[1];

      int mask_varid = 0;
      e_netcdf(nc_inq_varid(mask_ncid, "mask", &mask_varid));
      
      e_netcdf(nc_get_vara_double(mask_ncid, mask_varid, istart, icount, mask));
      int i = 0;
      for(i = 0; i < WIDTH; i++)
	mask[(HEIGHT - 1) * WIDTH + i] = 0;

      for(i = 0; i < len; i++)
      {
	e_netcdf(nc_get_vara_double(ncid, varid, lstart, lcount, buf));
        unpackDoubleInto(buf, mask, node + i * offset, sector, scale_factor, add_offset, bounds[0], bounds[1]);
        lstart[0]--;
      }

      safeFree(buf);
      safeFree(mask);
    } else
    {
      short * buf = (short *) calloc(GRID_SIZE, sizeof(short));

      e_netcdf(nc_get_att_float(ncid, varid, "scale_factor", &scale_factor));
      e_netcdf(nc_get_att_float(ncid, varid, "add_offset", &add_offset));

      e_netcdf(nc_get_att_float(ncid, varid, "unpacked_valid_range", bounds));

      var_min[sector] = bounds[0];
      var_max[sector] = bounds[1];

      int i = 0;
      for(i = 0; i < len; i++)
      {
	e_netcdf(nc_get_vara_short(ncid, varid, lstart, lcount, buf));
	unpackShortInto(buf, node + i * offset, sector, scale_factor, add_offset, bounds[0], bounds[1]);
	lstart[0]--;
      }

      safeFree(buf);
    }
  }

  void unpackLevel(int timeind, int ncid, int varid, int sector, int levind, float * node, int offset, int len)
  {
    size_t lstart[4] = {timeind, levind, 0, 0};
    size_t lcount[4] = {1, 1, HEIGHT, WIDTH};

    short * buf = (short *) calloc(GRID_SIZE, sizeof(short));
    float scale_factor = 0;
    float add_offset = 0;
    float bounds[2];

    e_netcdf(nc_get_att_float(ncid, varid, "scale_factor", &scale_factor));
    e_netcdf(nc_get_att_float(ncid, varid, "add_offset", &add_offset));

    e_netcdf(nc_get_att_float(ncid, varid, "unpacked_valid_range", bounds));

    var_min[sector] = bounds[0];
    var_max[sector] = bounds[1];

    int i = 0;
    for(i = 0; i < len; i++)
    {
      e_netcdf(nc_get_vara_short(ncid, varid, lstart, lcount, buf));
      unpackShortInto(buf, node + i * offset, sector, scale_factor, add_offset, bounds[0], bounds[1]);
      lstart[0]--;
    }

    safeFree(buf);
  }

  void unpackLevels(int timeind, int ncid, int varid, int sector, \
		    float * _n1000, float * _n925, float * _n850, float * _n700, float * _n500, float * _n250, int len)
  {
    unpackLevel(timeind, ncid, varid, sector, 0, _n1000, 7 * GRID_SIZE, len);
    unpackLevel(timeind, ncid, varid, sector, 1, _n925, 5 * GRID_SIZE, len);
    unpackLevel(timeind, ncid, varid, sector, 2, _n850, 5 * GRID_SIZE, len);
    unpackLevel(timeind, ncid, varid, sector, 3, _n700, 5 * GRID_SIZE, len);
    unpackLevel(timeind, ncid, varid, sector, 5, _n500, 5 * GRID_SIZE, len);
    unpackLevel(timeind, ncid, varid, sector, 7, _n250, 5 * GRID_SIZE, len);
  }

  void closeData()
  {
    nc_close(hgt_ncid);
    nc_close(pw_ncid);
    nc_close(rh_ncid);
    nc_close(sst_ncid);
    nc_close(tmp_ncid);
    nc_close(uwnd_ncid);
    nc_close(vwnd_ncid);
    nc_close(mask_ncid);

    loginfo("Closed all NetCDF files\n");
  }

  void unpackShortInto(short * buf, float * node, int sector, float scale_factor, float add_offset, \
                       float min, float max)
  {
    int i = 0;
    for(i = 0; i < GRID_SIZE; i++)
      node[sector * GRID_SIZE + i] = ((((float) buf[i]) * scale_factor + add_offset) - min) / (max - min);
  }

  void unpackDoubleInto(double * buf, double * mask, float * node, int sector, float scale_factor, float add_offset, \
                        float min, float max)
  {
    int i = 0;
    for(i = 0; i < GRID_SIZE; i++)
      node[sector * GRID_SIZE + i] = (((((float) buf[i]) * scale_factor + add_offset) * mask[i]) - min) / (max - min);
  }

  char * getYear()
  {
    int year = 0;
    loginfo("Year (1982-2013, anything else to exit)? ");
    if(scanf("%d", &year) != 1)
      exit(EXIT_FAILURE);

    if(year < 1982 || year > 2013)
      exit(EXIT_FAILURE);

    char * buf_year = (char *) calloc(5, sizeof(char));
    snprintf(buf_year, 5, "%d", year);

    return buf_year;
  }

  void openTimes()
  {
    // Reused
    int time_varid = 0;
    int dimid[1];
    
    e_netcdf(nc_inq_varid(hgt_ncid, "time", &time_varid));
    e_netcdf(nc_inq_vardimid(hgt_ncid, time_varid, dimid));
    e_netcdf(nc_inq_dimlen(hgt_ncid, dimid[0], &timelen));
    
    atime = (double *) calloc(timelen, sizeof(double));
    

    e_netcdf(nc_inq_varid(sst_ncid, "time", &time_varid));
    e_netcdf(nc_inq_vardimid(sst_ncid, time_varid, dimid));
    e_netcdf(nc_inq_dimlen(sst_ncid, dimid[0], &stimelen));

    astime = (double *) calloc(stimelen, sizeof(double));
  }

  void closeTimes()
  {
    safeFree(atime);
    safeFree(astime);
  }

  __global__ void updateNode(int level, int sector, float * weights, float * node, float * bottom, float * top, \
                             int sbot, int smid, int stop, int bnoffset, int noffset, int tnoffset, \
                             int bindex, int mindex, int tindex, int * NODE_OFFSETS, int * WEIGHT_OFFSETS)
  {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= GRID_SIZE) return;
    int i = 0, j = 0, y = n / 144, x = n % 144;
    int x1 = 0, y1 = 0, z1 = 0;

    int mul = (level == 1 ? SAMPLE_SIZE : 1);
    int woffset = sbot + smid + stop;

    int weightind = sector * (GRID_SIZE * (sbot + smid + stop)) * mul \
		  + y * (WIDTH * (sbot + smid + stop)) * mul \
		  + x * (sbot + smid + stop) * mul;
    int sizes[3] = {sbot, smid, stop};
    int noffsets[3] = {bnoffset, noffset, tnoffset};
    float * nodes[3] = {bottom, node, top};
    int aindices[3] = {bindex, mindex, tindex};

    int nodeind = sector * GRID_SIZE + y * WIDTH + x;
    float total = 0;

    // printf("%d %d %d %d %d\n", level, sector, x, y, noffset);

    // Handle wrap
    int subtotal = 0;
    for(i = 0; i < 3; i++)
      if(sizes[i] != 0)
      {
        for(x1 = 0; x1 < sizes[i] / 9; x1++)
          for(y1 = 0; y1 < 3; y1++)
            for(z1 = 0; z1 < 3; z1++)
              for(j = 0; j < mul; j++)
                total += weights[WEIGHT_OFFSETS[mindex * WEIGHT_OFFSET_BLOCK + level - 1] + j * woffset + weightind \
                                 + subtotal + x1 * 9 + y1 * 3 + z1] \
                       * nodes[i][NODE_OFFSETS[aindices[i] * NODE_OFFSET_BLOCK + level - 1] + j * noffsets[i] \
                                  + x1 * GRID_SIZE + wrap((y + (y1 - 1)), 0, HEIGHT) * WIDTH \
                       + wrap((x + (z1 - 1)), 0, WIDTH)];
        subtotal += sizes[i];
      }
    node[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] + nodeind] = sigmoid(total, (level == 1 ? (45 * SAMPLE_SIZE) : 45));
  }

  void updateNodes()
  {
    int i = 0, l = 0;
    int nblocks = (GRID_SIZE + 255) / 256;
    
    for(i = 1; i < HIDDEN_LAYERS + 2; i++)
    {
      for(l = 0; l < 7; l++)
        updateNode<<<nblocks, 256>>>(i, l, gw1000, gn1000, NULL, gn925, \
                                     WEIGHT_SURF_NUM_1, WEIGHT_SURF_NUM_2, WEIGHT_SURF_NUM_3, 0, \
                                     7 * GRID_SIZE, 5 * GRID_SIZE, 999, 0, 1, NODE_OFFSETS, WEIGHT_OFFSETS);
      for(l = 0; l < 5; l++)
      {
        updateNode<<<nblocks, 256>>>(i, l, gw925, gn925, gn1000, gn850, \
                                     WEIGHT_CSUR_NUM_1, WEIGHT_CSUR_NUM_2, WEIGHT_CSUR_NUM_3, \
                                     7 * GRID_SIZE, 5 * GRID_SIZE, 5 * GRID_SIZE, 0, 1, 2, NODE_OFFSETS, WEIGHT_OFFSETS);
        updateNode<<<nblocks, 256>>>(i, l, gw850, gn850, gn925, gn700, \
                                     WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, \
                                     5 * GRID_SIZE, 5 * GRID_SIZE, 5 * GRID_SIZE, 1, 2, 3, NODE_OFFSETS, WEIGHT_OFFSETS);
        updateNode<<<nblocks, 256>>>(i, l, gw700, gn700, gn850, gn500, \
                                     WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, \
                                     5 * GRID_SIZE, 5 * GRID_SIZE, 5 * GRID_SIZE, 2, 3, 4, NODE_OFFSETS, WEIGHT_OFFSETS);
        updateNode<<<nblocks, 256>>>(i, l, gw500, gn500, gn700, gn250, \
                                     WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, \
                                     5 * GRID_SIZE, 5 * GRID_SIZE, 5 * GRID_SIZE, 3, 4, 5, NODE_OFFSETS, WEIGHT_OFFSETS);
        updateNode<<<nblocks, 256>>>(i, l, gw250, gn250, gn500, NULL, \
                                     WEIGHT_TOP_NUM_1, WEIGHT_TOP_NUM_2, WEIGHT_TOP_NUM_3, \
                                     5 * GRID_SIZE, 5 * GRID_SIZE, 0, 4, 5, 999, NODE_OFFSETS, WEIGHT_OFFSETS);
      }
      cudaSync();
    }
  }

  __global__ void updateError(int level, int sector, float * weight, float * wbottom, float * wtop, \
                              float * node, float * actual, \
		              float * error, float * ebottom, float * etop, int sbot, int smid, int stop, \
		              int cbot, int ctop, int abot, int atop, bool isOutput, int noffset, \
                              int bindex, int mindex, int tindex, int * NODE_OFFSETS, int * WEIGHT_OFFSETS)
  {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= GRID_SIZE) return;

    int i = n / 144, j = n % 144, k = 0, l = 0;
    int sizes[3] = {sbot, smid, stop};
    int chunks[3] = {cbot, sbot + smid + stop, ctop};
    int add[3] = {abot, sbot, atop};
    float * errors[3] = {ebottom, error, etop};
    float * weights[3] = {wbottom, weight, wtop};
    int aindices[3] = {bindex, mindex, tindex};

    if(isOutput)
    {
      const int index = sector * GRID_SIZE + i * WIDTH + j;
      error[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] + index] = node[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] \
                                                                            + index] \
                                                                      * (1 - node[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] \
                                                                            + index]) \
                                                                      * (actual[index] \
                                                                      - node[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] \
                                                                            + index]);
    }
    else
    {
      float total = 0;
      const int index = sector * GRID_SIZE; // level
      for(l = 0; l < (level == 0 ? SAMPLE_SIZE : 1); l++)
      {
	for(k = 0; k < 3; k++)
	{
	  // Handle wrap
	  if(sizes[k] != 0)
	  {
	    int x1 = 0, y1 = 0, z1 = 0;
	    for(x1 = 0; x1 < sizes[k] / 9; x1++)
	      for(y1 = 0; y1 < 3; y1++)
		for(z1 = 0; z1 < 3; z1++)
		{
		  const int windex = x1 * (GRID_SIZE * chunks[k]) \
			       + wrap(i + (y1 - 1), 0, HEIGHT) * (WIDTH * chunks[k]) \
			       + wrap(j + (z1 - 1), 0, WIDTH) * chunks[k]; // level
		  const int eindex = x1 * GRID_SIZE; // level + 1
	          total += errors[k][NODE_OFFSETS[aindices[k] * NODE_OFFSET_BLOCK + level + 1] + eindex \
                                     + wrap(i + (y1 - 1), 0, HEIGHT) * WIDTH + wrap(j + (z1 - 1), 0, WIDTH)] \
                         * weights[k][WEIGHT_OFFSETS[aindices[k] * WEIGHT_OFFSET_BLOCK + level] \
                                      + l * chunks[k] + windex + add[k] + sector * 9 + (2 - y1) * 3 + (2 - z1)];
		}
	  }
	}
	error[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] \
              + l * noffset + index + i * WIDTH + j] = node[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] \
                                                            + l * noffset + index + i * WIDTH + j] \
              * (1 - node[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] + l * noffset + index + i * WIDTH + j]) * total;
      }
    }
  }

  void updateErrors()
  {

    // simple
    int i = 0, j = 0;
    int nblocks = (GRID_SIZE + 255) / 256;

    for(i = 0; i < 7; i++)
      updateError<<<nblocks, 256>>>(HIDDEN_LAYERS + 1, i, gw1000, NULL, gw925, gn1000, gnn1000, ge1000, NULL, ge925, \
                                    0, 0, 0, 0, 0, 0, 0, true, 7 * GRID_SIZE, 999, 0, 1, NODE_OFFSETS, WEIGHT_OFFSETS);
    for(i = 0; i < 5; i++)
    {
      updateError<<<nblocks, 256>>>(HIDDEN_LAYERS + 1, i, gw925, gw1000, gw850, gn925, gnn925, ge925, ge1000, ge850, \
                                    0, 0, 0, 0, 0, 0, 0, true, 5 * GRID_SIZE, 0, 1, 2, NODE_OFFSETS, WEIGHT_OFFSETS);
      updateError<<<nblocks, 256>>>(HIDDEN_LAYERS + 1, i, gw850, gw925, gw700, gn850, gnn850, ge850, ge925, ge700, \
                                    0, 0, 0, 0, 0, 0, 0, true, 5 * GRID_SIZE, 1, 2, 3, NODE_OFFSETS, WEIGHT_OFFSETS);
      updateError<<<nblocks, 256>>>(HIDDEN_LAYERS + 1, i, gw700, gw850, gw500, gn700, gnn700, ge700, ge850, ge500, \
                                    0, 0, 0, 0, 0, 0, 0, true, 5 * GRID_SIZE, 2, 3, 4, NODE_OFFSETS, WEIGHT_OFFSETS);
      updateError<<<nblocks, 256>>>(HIDDEN_LAYERS + 1, i, gw500, gw700, gw250, gn500, gnn500, ge500, ge700, ge250, \
                                    0, 0, 0, 0, 0, 0, 0, true, 5 * GRID_SIZE, 3, 4, 5, NODE_OFFSETS, WEIGHT_OFFSETS);
      updateError<<<nblocks, 256>>>(HIDDEN_LAYERS + 1, i, gw250, gw500, NULL, gn250, gnn250, ge250, ge500, NULL, \
                                    0, 0, 0, 0, 0, 0, 0, true, 5 * GRID_SIZE, 4, 5, 999, NODE_OFFSETS, WEIGHT_OFFSETS);
    }

    cudaSync();

    // complex
    for(j = HIDDEN_LAYERS; j > -1; j--)
    {
      for(i = 0; i < 7; i++)
	updateError<<<nblocks, 256>>>(j, i, gw1000, NULL, gw925, gn1000, NULL, ge1000, NULL, ge925, \
                    WEIGHT_SURF_NUM_1, WEIGHT_SURF_NUM_2, WEIGHT_SURF_NUM_3, \
		    0, WEIGHT_CSUR_NUM_1 + WEIGHT_CSUR_NUM_2 + WEIGHT_CSUR_NUM_3, 0, \
                    0, false, 7 * GRID_SIZE, 999, 0, 1, NODE_OFFSETS, WEIGHT_OFFSETS);
      for(i = 0; i < 5; i++)
      {
	updateError<<<nblocks, 256>>>(j, i, gw925, gw1000, gw850, gn925, NULL, ge925, ge1000, ge850, \
                    WEIGHT_CSUR_NUM_1, WEIGHT_CSUR_NUM_2, WEIGHT_CSUR_NUM_3, \
		    WEIGHT_SURF_NUM_1 + WEIGHT_SURF_NUM_2 + WEIGHT_SURF_NUM_3, \
		    WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2 + WEIGHT_BODY_NUM_3, WEIGHT_SURF_NUM_1 + WEIGHT_SURF_NUM_2, \
                    0, false, 5 * GRID_SIZE, 0, 1, 2, NODE_OFFSETS, WEIGHT_OFFSETS);
	updateError<<<nblocks, 256>>>(j, i, gw850, gw925, gw700, gn850, NULL, ge850, ge925, ge700, \
                    WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, \
		    WEIGHT_CSUR_NUM_1 + WEIGHT_CSUR_NUM_2 + WEIGHT_CSUR_NUM_3, \
		    WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2 + WEIGHT_BODY_NUM_3, WEIGHT_CSUR_NUM_1 + WEIGHT_CSUR_NUM_2, \
                    0, false, 5 * GRID_SIZE, 1, 2, 3, NODE_OFFSETS, WEIGHT_OFFSETS);
	updateError<<<nblocks, 256>>>(j, i, gw700, gw850, gw500, gn700, NULL, ge700, ge850, ge500, \
                    WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, \
		    WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2 + WEIGHT_BODY_NUM_3, \
		    WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2 + WEIGHT_BODY_NUM_3, WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2, \
                    0, false, 5 * GRID_SIZE, 2, 3, 4, NODE_OFFSETS, WEIGHT_OFFSETS);
	updateError<<<nblocks, 256>>>(j, i, gw500, gw700, gw250, gn500, NULL, ge500, ge700, ge250, \
                    WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, \
		    WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2 + WEIGHT_BODY_NUM_3, \
		    WEIGHT_TOP_NUM_1 + WEIGHT_TOP_NUM_2 + WEIGHT_TOP_NUM_3, WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2, \
                    0, false, 5 * GRID_SIZE, 3, 4, 5, NODE_OFFSETS, WEIGHT_OFFSETS);
	updateError<<<nblocks, 256>>>(j, i, gw250, gw500, NULL, gn250, NULL, ge250, ge500, NULL, \
                    WEIGHT_TOP_NUM_1, WEIGHT_TOP_NUM_2, WEIGHT_TOP_NUM_3, \
		    WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2 + WEIGHT_BODY_NUM_3, 0, WEIGHT_BODY_NUM_1 + WEIGHT_BODY_NUM_2, \
                    0, false, 5 * GRID_SIZE, 4, 5, 999, NODE_OFFSETS, WEIGHT_OFFSETS);
      }
      cudaSync();
    }
  }
  
  // Backpropagation
  __global__ void updateWeight(int level, int sector, float * weight, \
		    float * node, float * bnode, float * tnode, float * error, int sbot, int smid, int stop, int noffset, \
                    int bindex, int mindex, int tindex, int * NODE_OFFSETS, int * WEIGHT_OFFSETS)
  {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= GRID_SIZE) return;
    int i = n / 144, j = n % 144, k = 0, l = 0;
    int sizes[3] = {sbot, smid, stop};
    float * nodes[3] = {bnode, node, tnode};
    int aindices[3] = {bindex, mindex, tindex};

    int woffset = sbot + smid + stop;  

    int subtotal = 0;
    for(k = 0; k < 3; k++)
      if(sizes[k] != 0)
      {
        int x1 = 0, y1 = 0, z1 = 0;
        for(x1 = 0; x1 < sizes[k] / 9; x1++)
          for(y1 = 0; y1 < 3; y1++)
            for(z1 = 0; z1 < 3; z1++)
              for(l = 0; l < (level == 1 ? SAMPLE_SIZE : 1); l++)
	      {
		const int windex = sector * (GRID_SIZE * (sbot + smid + stop)) + \
                                   i * (WIDTH * (sbot + smid + stop)) + \
                                   j * (sbot + smid + stop) + \
                                   subtotal + x1 * 9 + y1 * 3 + z1;
		const int eindex = sector * GRID_SIZE + i * WIDTH + j;
		const int nindex = x1 * GRID_SIZE + wrap(i + (y1 - 1), 0, HEIGHT) *  WIDTH + wrap(j + (z1 - 1), 0, WIDTH);
		weight[WEIGHT_OFFSETS[mindex * WEIGHT_OFFSET_BLOCK + level - 1] \
                       + l * woffset + windex] += LEARNING_RATE * error[NODE_OFFSETS[mindex * NODE_OFFSET_BLOCK + level] \
                                                                        + l * noffset + eindex] \
                                                * nodes[k][NODE_OFFSETS[aindices[k] * NODE_OFFSET_BLOCK + level - 1] \
                                                                        + l * noffset + nindex];
	      }
        subtotal += sizes[k];
      }
  }

  void updateWeights()
  {
    int i = 0, j = 0;
    int nblocks = (GRID_SIZE + 255) / 256;
    for (j = HIDDEN_LAYERS + 1; j > 0; j--)
    {
      for(i = 0; i < 7; i++)
	updateWeight<<<nblocks, 256>>>(j, i, gw1000, gn1000, NULL, gn925, ge1000, \
                                       WEIGHT_SURF_NUM_1, WEIGHT_SURF_NUM_2, WEIGHT_SURF_NUM_3, 7 * GRID_SIZE, 999, 0, 1, \
                                       NODE_OFFSETS, WEIGHT_OFFSETS);
      for(i = 0; i < 5; i++)
      {
	updateWeight<<<nblocks, 256>>>(j, i, gw925, gn925, gn1000, gn850, ge925, \
                                       WEIGHT_CSUR_NUM_1, WEIGHT_CSUR_NUM_2, WEIGHT_CSUR_NUM_3, 5 * GRID_SIZE, 0, 1, 2, \
                                       NODE_OFFSETS, WEIGHT_OFFSETS);
	updateWeight<<<nblocks, 256>>>(j, i, gw850, gn850, gn925, gn700, ge850, \
                                       WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, 5 * GRID_SIZE, 1, 2, 3, \
                                       NODE_OFFSETS, WEIGHT_OFFSETS);
	updateWeight<<<nblocks, 256>>>(j, i, gw700, gn700, gn850, gn500, ge700, \
                                       WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, 5 * GRID_SIZE, 2, 3, 4, \
                                       NODE_OFFSETS, WEIGHT_OFFSETS);
	updateWeight<<<nblocks, 256>>>(j, i, gw500, gn500, gn700, gn250, ge500, \
                                       WEIGHT_BODY_NUM_1, WEIGHT_BODY_NUM_2, WEIGHT_BODY_NUM_3, 5 * GRID_SIZE, 3, 4, 5, \
                                       NODE_OFFSETS, WEIGHT_OFFSETS);
	updateWeight<<<nblocks, 256>>>(j, i, gw250, gn250, gn500, NULL, ge250, \
                                       WEIGHT_TOP_NUM_1, WEIGHT_TOP_NUM_2, WEIGHT_TOP_NUM_3, 5 * GRID_SIZE, 4, 5, 6, \
                                       NODE_OFFSETS, WEIGHT_OFFSETS);
      }
      cudaSync();
    }
  }

  void loopNodes()
  {
    float * nodes[6] = {n1000, n925, n850, n700, n500, n250};
    int sectors[6] = {7, 5, 5, 5, 5, 5};
    int i = 0;
    
    for(i = 0; i < 6; i++)
    {
      memmove(nodes[i] + sectors[i] * GRID_SIZE, nodes[i], sectors[i] * GRID_SIZE * (SAMPLE_SIZE - 1) * sizeof(float));
      memcpy(nodes[i], nodes[i] + H_NODE_OFFSETS[i * NODE_OFFSET_BLOCK + HIDDEN_LAYERS + 1], \
             sectors[i] * GRID_SIZE * sizeof(float));
    }
  }

  __global__ void getOutputErrorSingle(float * error, int * NODE_OFFSETS, int sectorCount, int eindex)
  {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n > 0) return;

    int i = 0, j = 0, k = 0;
    float total = 0;
    for(i = 0; i < sectorCount; i++)
      for(j = 0; j < HEIGHT; j++)
	for(k = 0; k < WIDTH; k++)
	  total += powf(error[NODE_OFFSETS[eindex * NODE_OFFSET_BLOCK + HIDDEN_LAYERS + 1] + i * GRID_SIZE + j * WIDTH + k], 2.0f);
    toterr += total;
  }
  
  float getOutputError()
  {
    float * errors[6] = {ge1000, ge925, ge850, ge700, ge500, ge250};
    int sectors[6] = {7, 5, 5, 5, 5, 5};
    int i = 0;
    float total = 0;

    cudaMemcpyToSymbol(toterr, &total, sizeof(float), 0, cudaMemcpyHostToDevice);

    for(i = 0; i < 6; i++)
      getOutputErrorSingle<<<1, 1>>>(errors[i], NODE_OFFSETS, sectors[i], i);

    cudaSync();
    cudaMemcpyFromSymbol(&total, toterr, sizeof(float), 0, cudaMemcpyDeviceToHost);
    cudaSync();

    return 0.5f * total;
  }

  void writeNetCDF(char * year, int index, int timelapse)
  {
    int i = 0, j = 0;
    char buf[256];
    int forecastncid, varids[10], datadimids[3], leveldimids[1], londimids[1], latdimids[1];
    float * nodes[6] = {n1000, n925, n850, n700, n500, n250};
    float lengths[6] = {7, 5, 5, 5, 5, 5};
    char * datanames[] = {"air_temperature", "heights", "rel_humidity", "u_wind", "v_wind", "sst", "pr_water"};

    snprintf(buf, 255, "../forecast/forecast.%.4s.%d.%03dhr.nc", year, index, timelapse);

    e_netcdf(nc_create(buf, NC_CLOBBER, &forecastncid));
    loginfo("Created NetCDF file...\n");
    e_netcdf(nc_def_dim(forecastncid, "level", 6, &datadimids[0]));
    e_netcdf(nc_def_dim(forecastncid, "lat", 73, &datadimids[1]));
    e_netcdf(nc_def_dim(forecastncid, "lon", 144, &datadimids[2]));
    loginfo("Created dimensions...\n");

    londimids[0] = datadimids[2];
    latdimids[0] = datadimids[1];
    leveldimids[0] = datadimids[0];
  
    for(i = 0; i < 7; i++)
      e_netcdf(nc_def_var(forecastncid, datanames[i], NC_FLOAT, 3, datadimids, &varids[i]));

    e_netcdf(nc_def_var(forecastncid, "level", NC_FLOAT, 1, leveldimids, &varids[7]));
    e_netcdf(nc_def_var(forecastncid, "lon", NC_FLOAT, 1, londimids, &varids[8]));
    e_netcdf(nc_def_var(forecastncid, "lat", NC_FLOAT, 1, latdimids, &varids[9]));
    
    loginfo("Defined all vars...\n");
  
    e_netcdf(nc_put_att_text(forecastncid, varids[7], "units", 8, "millibar"));
    e_netcdf(nc_put_att_text(forecastncid, varids[8], "units", 11, "degree_east"));
    e_netcdf(nc_put_att_text(forecastncid, varids[9], "units", 12, "degree_north"));

    e_netcdf(nc_put_att_text(forecastncid, varids[7], "long_name", 5, "Level"));
    e_netcdf(nc_put_att_text(forecastncid, varids[8], "long_name", 9, "Longitude"));
    e_netcdf(nc_put_att_text(forecastncid, varids[9], "long_name", 8, "Latitude"));

    e_netcdf(nc_put_att_text(forecastncid, varids[7], "axis", 1, "Z"));
    e_netcdf(nc_put_att_text(forecastncid, varids[8], "axis", 1, "X"));
    e_netcdf(nc_put_att_text(forecastncid, varids[9], "axis", 1, "Y"));

    e_netcdf(nc_put_att_text(forecastncid, varids[7], "positive", 4, "down"));

    const char * units[] = {"degK", "m", "%", "m/s", "m/s", "degC", "kg/m^2"};
    for(i = 0; i < 7; i++)
      e_netcdf(nc_put_att_text(forecastncid, varids[i], "units", strlen(units[i]), units[i]));

    loginfo("Defined all attributes..\n");
 
    e_netcdf(nc_enddef(forecastncid));
    
    // Begin definition for dimensions
    float levels[6] = {1000.0f, 925.0f, 850.0f, 700.0f, 500.0f, 300.0f};
    float lon[144] = {0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, \
                      30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5, \
                      60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, \
                      90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110, 112.5, 115, 117.5, \
                      120, 122.5, 125, 127.5, 130, 132.5, 135, 137.5, 140, 142.5, 145, 147.5, 150, \
                      152.5, 155, 157.5, 160, 162.5, 165, 167.5, 170, 172.5, 175, 177.5, \
                      180, 182.5, 185, 187.5, 190, 192.5, 195, 197.5, 200, 202.5, 205, 207.5, \
                      210, 212.5, 215, 217.5, 220, 222.5, 225, 227.5, 230, 232.5, 235, 237.5, \
                      240, 242.5, 245, 247.5, 250, 252.5, 255, 257.5, 260, 262.5, 265, 267.5, \
                      270, 272.5, 275, 277.5, 280, 282.5, 285, 287.5, 290, 292.5, 295, 297.5, \
                      300, 302.5, 305, 307.5, 310, 312.5, 315, 317.5, 320, 322.5, 325, 327.5, \
                      330, 332.5, 335, 337.5, 340, 342.5, 345, 347.5, 350, 352.5, 355, 357.5};
    float lat[73] = {90,  87.5,  85,  82.5,  80,  77.5,  75,  72.5,  70,  67.5,  65,  62.5,  \
                     60, 57.5, 55, 52.5, 50, 47.5, 45, 42.5, 40, 37.5, 35, 32.5, \
                     30, 27.5, 25, 22.5, 20, 17.5, 15, 12.5, 10, 7.5, 5, 2.5, \
                     0, -2.5, -5, -7.5, -10, -12.5, -15, -17.5, -20, -22.5, -25, -27.5, \
                     -30, -32.5, -35, -37.5, -40, -42.5, -45, -47.5, -50, -52.5, -55, -57.5, \
                     -60, -62.5, -65, -67.5, -70, -72.5, -75, -77.5, -80, -82.5, -85, -87.5, -90};

    nc_put_var_float(forecastncid, varids[7], levels);
    nc_put_var_float(forecastncid, varids[8], lon);
    nc_put_var_float(forecastncid, varids[9], lat);

    const size_t count[] = {1, 73, 144};
   
    int k = 0, l = 0;
    for(i = 0; i < 6; i++)
    {
      const size_t start[] = {i, 0, 0};
      for(j = 0; j < lengths[i]; j++)
      {
        float scaled[GRID_SIZE];
        for(k = 0; k < HEIGHT; k++)
          for(l = 0; l < WIDTH; l++)
            scaled[k * WIDTH + l] = nodes[i][H_NODE_OFFSETS[i * NODE_OFFSET_BLOCK + HIDDEN_LAYERS + 1] \
                                             + j * GRID_SIZE + k * WIDTH + l] * (var_max[j] - var_min[j]) + var_min[j];
        e_netcdf(nc_put_vara_float(forecastncid, varids[j], start, count, scaled));
      }
    }

    loginfo("Put all vars...\n"); 

    e_netcdf(nc_close(forecastncid));
    loginfo("Closed NetCDF file...\n");
  }

  void displayNetwork(int sector, int layer, int level, float * network)
  {
    int i = 0, j = 0;
    char * ascii = "*WMB8&%$#@oahkbdpqwmLCJUYXZO0Qrcvunxzjft/\\|()1{}[]-_+<>i!lI?.'`,^:\";~";
    float stval = 0.0f, endval = 1.0f;

    for(i = 0; i < HEIGHT; i++)
    {
      for(j = 0; j < WIDTH; j++)
      {
        float value = network[H_NODE_OFFSETS[level * NODE_OFFSET_BLOCK + layer] + sector * GRID_SIZE + i * WIDTH + j];
        int index = floor((value - stval) / (endval - stval) * (strlen(ascii) - 1));
        if(value >= stval && value <= endval)
          printf("%.1s", ascii + index);
        else
          printf("%s", " ");
      }
      printf("%s", "\n");
    }    
  }

  // Assume n and nn nodes are already read into properly
  void checkNetworkSanity(int level, int sector, int layer)
  {
    float * nodes[6] = {n1000, n925, n850, n700, n500, n250};
    displayNetwork(sector, layer, level, nodes[level]);
  }

  void train(char * year)
  {
    _INIT();

    alloc();
    openNetwork();

    readNetwork(); 

    /*
    int reCount = 0;
    const int COUNT_THRESHOLD = 10000;
    const float ERROR_THRESHOLD = 0.00001f;
    float outputError = 0; 
    do
    {
     int i = 0, j = 0, k = 0;
     int sectors[6] = {7, 5, 5, 5, 5, 5};
     float * nodes[6] = {n1000, n925, n850, n700, n500, n250};
     float * next[6] = {nn1000, nn925, nn850, nn700, nn500, nn250};
     char buf[128];

     const int REPEAT = 8;

     for(i = 0; i < 3; i++)
       for(j = 0; j < sectors[i] * GRID_SIZE * SAMPLE_SIZE; j++)
	 nodes[i][j] = 1.0f;
     for(i = 3; i < 6; i++)
       for(j = 0; j < sectors[i] * GRID_SIZE * SAMPLE_SIZE; j++)
	 nodes[i][j] = 0.0f;
     for(i = 0; i < 6; i++)
       for(j = 0; j < sectors[i] * GRID_SIZE; j++)
	 next[i][j] = 1.0f;
     
     cudaRead();
     
     for(k = 0; k < REPEAT; k++)
     {
       updateNodes();
       loginfo("Update nodes\n");
       updateErrors();
       loginfo("Update errors\n");
       updateWeights();
       loginfo("Update weights\n");
       
       outputError = getOutputError();
       snprintf(buf, 127, "Total error: %f\n", outputError);
       loginfo(buf);
     }     

     cudaWrite(); 
	    
     for(i = 0; i < 3; i++)
       for(j = 0; j < sectors[i] * GRID_SIZE * SAMPLE_SIZE; j++)
	 nodes[i][j] = 0.0f;
     for(i = 3; i < 6; i++)
       for(j = 0; j < sectors[i] * GRID_SIZE * SAMPLE_SIZE; j++)
	 nodes[i][j] = 1.0f;
     for(i = 0; i < 6; i++)
       for(j = 0; j < sectors[i] * GRID_SIZE; j++)
	 next[i][j] = 0.0f;

     cudaRead();

     for(k = 0; k < REPEAT; k++)
     {     
       updateNodes();
       loginfo("Update nodes\n");
       updateErrors();
       loginfo("Update errors\n");
       updateWeights();
       loginfo("Update weights\n");
       
       outputError = getOutputError();
       snprintf(buf, 127, "Total error: %f\n", outputError);
       loginfo(buf);
     }

     cudaWrite(); 
    } while (reCount < COUNT_THRESHOLD && outputError > ERROR_THRESHOLD);
    */

    //
    loginfo("Reading time information...\n");
    char * buf_year = (strcmp(year, "-1") == 0 ? getYear() : year);
    openData(buf_year);
    safeFree(buf_year);

    loginfo("Reading time files...\n");
    openTimes();

    int i = 0, j = 0, k = 0, l = 0;

    debug = fopen("outputs.txt", "a");

    int reCount = 0;
    const int COUNT_THRESHOLD = 10000;
    const float ERROR_THRESHOLD = 0.00001f;
    float outputError = 0; 
    int last = 0;
    do
    {
      j = 0;

      for(k = SAMPLE_SIZE - 1; k < 16; k++) //timelen - 1 - SAMPLE_SIZE; i++)
      {
        last = i;
        while(i == last)
          i = randrange(SAMPLE_SIZE - 1, 16);

        loginfo("Training index: ");
        printf("%d\n", i);

	char buf[128];
	snprintf(buf, 127, "%.3f percent of time trained: %d/%d\n", \
		 ((float) k) / (float) (timelen - 1) * 100, (int) k, (int) timelen);
	loginfo(buf);    

	while(j < stimelen - 1 && astime[j] <= atime[i])
	  ++j;

	readData(i, j, n1000, n925, n850, n700, n500, n250, SAMPLE_SIZE);
	loginfo("Read data\n");
	readData(i + 1, j, nn1000, nn925, nn850, nn700, nn500, nn250, 1);
	loginfo("Checking data\n");

        cudaRead();

        for(l = 0; l < 10; l++)
        {
	  updateNodes();
	  loginfo("Update nodes\n");
	  updateErrors();
	  loginfo("Update errors\n");
	  updateWeights();
	  loginfo("Update weights\n");

	  outputError = getOutputError();
	  snprintf(buf, 127, "Total error: %f\n", outputError);
	  loginfo(buf);

          loginfo(" -- Updated ");
          printf("%d times --\n", l);
        } 
      
        cudaWrite();
      }

      reCount++;

      writeNetwork(); 

      char rebuf[128];
      snprintf(rebuf, 127, "** Loop Count: %d **\n", reCount);
      loginfo(rebuf);
    } while (reCount < COUNT_THRESHOLD && outputError > ERROR_THRESHOLD);

    fclose(debug);

    closeTimes();

    closeData();
    //   
    closeNetwork();
    dealloc();

    cudaError(cudaDeviceReset());
  }

  void forecast(char * year, int index)
  {
    _INIT();

    alloc();
    openNetwork();

    readNetwork();

    char * buf_year = (strcmp(year, "-1") == 0 ? getYear() : year);
    openData(buf_year);

    loginfo("Reading time files...\n");
    openTimes();

    if(index < SAMPLE_SIZE - 1 || index >= timelen)
    {
      logerr("Invalid input with index\n");
      exit(EXIT_FAILURE);
    }

    int j = 0;

    while(j < stimelen - 1 && astime[j] <= atime[index])
      ++j;

    int timelapse = 6;

    readData(index, j, n1000, n925, n850, n700, n500, n250, SAMPLE_SIZE);
    loginfo("Read data\n");

// ---
    char buf[128];
    readData(index + 1, j, nn1000, nn925, nn850, nn700, nn500, nn250, 1);
//

    cudaRead();

    updateNodes();
    loginfo("Update nodes\n");

// ---
    updateErrors();
//

    cudaWrite();

// ---
    float outputError = getOutputError();
    snprintf(buf, 127, "Total error: %f\n", outputError);
    loginfo(buf);
//

    checkNetworkSanity(0, 0, HIDDEN_LAYERS + 1);

    writeNetCDF(buf_year, index, timelapse);
    loginfo("Write NetCDF data ");
    printf("%s yr %d index %d hr\n", buf_year, index, timelapse);

    for(timelapse = 12; timelapse <= 384; timelapse += 6)
    {
      loopNodes();
      loginfo("Loopback data\n");

      cudaRead();

      updateNodes();
      loginfo("Update nodes\n");
      
      cudaWrite();

      writeNetCDF(buf_year, index, timelapse); 
      loginfo("Write NetCDF data\n");
      printf("%s yr %d index %d hr\n", year, index, timelapse);
    }

    safeFree(buf_year);
    closeTimes();

    closeData();
    
    closeNetwork();
    dealloc();

    cudaError(cudaDeviceReset());
  }

  inline void cudaSync()
  {
    cudaError(cudaDeviceSynchronize());
  }

  void cudaError(cudaError_t err)
  {
    if(err != cudaSuccess)
    {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
      exit(1);
      // leak on purpose
    }
  }

  void safeFree(void * ptr)
  {
    free(ptr);
    ptr = NULL;
  }
}

