/*
 * =====================================================================================
 *
 *       Filename:  xor.h
 *
 *    Description:  Header file for xor.cu
 *
 *        Version:  1.0
 *        Created:  02/02/2014 05:48:51 PM
 *       Revision:  none
 *       Compiler:  gcc/nvcc
 *
 *         Author:  Jonathan Ni 
 *   Organization:  Productive Productions
 *
 * =====================================================================================
 */

#define HIDDEN_LAYERS 1

#define INPUT_NODES 2
#define HIDDEN_NODES 4
#define OUTPUT_NODES 1

struct neuron
{
  struct neuron ** prev;
  float * weights;
  float activity;
};

void alloc(void);
void dealloc(void);
