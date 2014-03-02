#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "xor.h"

struct neuron input[INPUT_NODES];
struct neuron hidden[HIDDEN_LAYERS][HIDDEN_NODES];
struct neuron output[OUTPUT_NODES];

int main(int argc, char * argv[])
{
  if(argc != 2) return 1;
  long choice = strtol(argv[1], NULL, 10);
  if(choice < 0 || choice > 1) return 1;

  alloc();

  if(choice == 0)
  {
  }
  else
  {
  }
  
  dealloc();

  return 0;
}

void alloc()
{
  int i = 0, j = 0;
  for(i = 0; i < INPUT_NODES; i++)
  {
    input[i].neuron = NULL;
    input[i].weights = NULL;
    input[i].activity = 0;
  }
  for(i = 0; i < HIDDEN_LAYERS; i++)
    for(j = 0; j < HIDDEN_NODES; j++)
    {
      if(i == 0)
      {
        hidden[i].neuron = (struct neuron **) calloc(INPUT_NODES, sizeof(struct neuron *));
        hidden[i].weights = (float *) calloc(INPUT_NODES, sizeof(float));
        hidden[i].activity = 0;

        int k = 0;
        for(k = 0; k < INPUT_NODES; k++)
          hidden[i].neuron[k] = &input[k];
      }
    }
  for(i = 0; i < OUTPUT_NODES; i++)
  {
    output[i].neuron = (struct neuron **) calloc(HIDDEN_NODES, sizeof(struct neuron *));
    output[i].weights = (float *) calloc(HIDDEN_NODES, sizeof(float));
    output[i].activity = 0;

    for(j = 0; j < HIDDEN_NODES; j++)
      output[i].neuron[j] = &hidden[k];
  }
}

void dealloc()
{

}
