/*
 * =====================================================================================
 *
 *       Filename:  xor.h
 *
 *    Description:  Header file for xor test
 *
 *        Version:  1.0
 *        Created:  03/01/2014 08:29:23 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#define NUM_INPUT 2
#define NUM_OUTPUT 1
#define NUM_LAYERS 3
#define NUM_NEURONS_HIDDEN 3

#define DESIRED_ERROR 0.0001f

#define MAX_EPOCHS 500000
#define EPOCHS_BETWEEN_REPORTS 1000

void train(void);
