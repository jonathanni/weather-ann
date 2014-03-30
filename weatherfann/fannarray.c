/*
 * =====================================================================================
 *
 *       Filename:  fannarray.c
 *
 *    Description:  array -> fann_train_data
 *
 *        Version:  1.0
 *        Created:  05/22/2007 10:35:00 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  dukhat
 *   Organization:  
 *
 * =====================================================================================
 */

#include <fann.h>
#include "include/fannarray.h"

struct fann_train_data *read_from_array(fann_type *din, fann_type *dout, unsigned int num_data,
unsigned int num_input, unsigned int num_output) {
  unsigned int i, j;
  fann_type *data_input, *data_output;
  struct fann_train_data *data = (struct fann_train_data *) malloc(sizeof(struct fann_train_data));
  if(data == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }
 
  fann_init_error_data((struct fann_error *) data);
 
  data->num_data = num_data;
  data->num_input = num_input;
  data->num_output = num_output;
  data->input = (fann_type **) calloc(num_data, sizeof(fann_type *));
  if(data->input == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }
 
  data->output = (fann_type **) calloc(num_data, sizeof(fann_type *));
  if(data->output == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }
 
  data_input = (fann_type *) calloc(num_input * num_data, sizeof(fann_type));
  if(data_input == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }
 
  data_output = (fann_type *) calloc(num_output * num_data, sizeof(fann_type));
  if(data_output == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }
 
  for(i = 0; i != num_data; i++) {
    data->input[i] = data_input;
    data_input += num_input;
   
    for(j = 0; j != num_input; j++) {
      data->input[i][j] = din[i*num_input+j];
    }
   
   
    data->output[i] = data_output;
    data_output += num_output;
   
    for(j = 0; j != num_output; j++) {
      data->output[i][j] = dout[i*num_output+j];
    }
  }
  return data;
}
