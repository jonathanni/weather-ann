#include "include/error.h"
#include "include/logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <string.h>

void e_netcdf(int errorcode)
{
  if(errorcode == NC_NOERR)
    return;
  char buf[128];

  strcpy(buf, (char *) nc_strerror(errorcode));
  strcat(buf, "\n");
  logerr(buf);

  exit(EXIT_FAILURE);
}
