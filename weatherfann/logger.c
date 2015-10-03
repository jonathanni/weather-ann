#include <stdio.h>
#include "include/logger.h"

void loginfo(char * str)
{
  fprintf(stdout, "[I] %s", str);
  fflush(stdout);
}

void logwarn(char * str)
{
  fprintf(stderr, "[W] %s", str);
}

void logerr(char * str)
{
  fprintf(stderr, "[E] %s", str);
}
