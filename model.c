#include "include/vars.h"
#include "include/logger.h"
#include "include/error.h"
#include "include/engine.h"

#include <netcdf.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int main(int argc, char * argv[])
{
  struct cudaDeviceProp prop;
  int deviceId = 0;
  char dbuf[128];

  setbuf(stdout, NULL);

  printf("%s\n", "Usage: model <0 - Train/1 - Forecast> <Year> <InternalIndex>");

  loginfo("Starting model...\n");
  if(system("clear") != 0) exit(EXIT_FAILURE);

  cudaError(cudaSetDevice(deviceId));
  cudaError(cudaGetDeviceProperties(&prop, deviceId));

  loginfo("CUDA device check...\n");
  snprintf(dbuf, 128, "Device: %s\n", prop.name);
  loginfo(dbuf);

  int choice, index;
  char * year = (char *) calloc(128, sizeof(char));
  strcpy(year, "-1");

  if(argc < 2)
  {  
    loginfo("Selection screen: \n");
    loginfo("0 for training, 1 for forecasting, or anything else to exit\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

    loginfo("model> ");

    if(scanf("%d", &choice) != 1)
    {
      logerr("Invalid input\n");
      exit(EXIT_FAILURE);
    }
  } else
    choice = (int) strtol(argv[1], &argv[1], 10);  

  if(argc >= 3)
    strcpy(year, argv[2]); 

  if(argc == 4)
    index = (int) strtol(argv[3], NULL, 10);
  else if(choice == 1)
  {
    loginfo("\nPick a internal index\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    
    loginfo("model> ");
    if(scanf("%d", &index) < 0)
    {
      exit(EXIT_FAILURE);
      logerr("Invalid input\n");
    }
  }

  if(choice == 0)
    train(year);
  else if(choice == 1)
    forecast(year, index);

  if(argc == 1) safeFree(year);
  return 0;
}
