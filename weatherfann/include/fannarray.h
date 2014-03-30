/*
 * =====================================================================================
 *
 *       Filename:  fannarray.h
 *
 *    Description:  arrays -> struct fann_train_data
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

#ifndef _FANN_ARRAY_GUARD
#define _FANN_ARRAY_GUARD

#include "fann.h"

struct fann_train_data *read_from_array(fann_type *, fann_type *, unsigned int, unsigned int, unsigned int);

#endif
