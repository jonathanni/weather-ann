/*
 * =====================================================================================
 *
 *       Filename:  xor.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/01/2014 08:29:06 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <fann.h>
#include "xor.h"

void train()
{
  struct fann *ann = fann_create_standard(NUM_LAYERS, NUM_INPUT, NUM_NEURONS_HIDDEN, NUM_OUTPUT);
  
  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  fann_train_on_file(ann, "xor.data", MAX_EPOCHS, EPOCHS_BETWEEN_REPORTS, DESIRED_ERROR);
  
  fann_save(ann, "xor_float.net");
}

void execute()
{
  fann_type *calc_out;
  fann_type input[2];

  struct fann *ann = fann_create_from_file("xor_float.net");

  input[0] = -1;
  input[1] = 1;

  calc_out = fann_run(ann, input);

  printf("xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);

  fann_destroy(ann);
}

int main()
{
  train();
  execute();
  return 0;
}
