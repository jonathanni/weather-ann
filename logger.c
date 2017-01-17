#include <stdio.h>
#include "include/logger.h"

loglevel_type loglevel = L_INFO;

void logdebug(char * str)
{
  if(loglevel > L_DEBUG) return;
  fprintf(stdout, "[D] %s", str);
  fflush(stdout);
}

void loginfo(char * str)
{
  if(loglevel > L_INFO) return;
  fprintf(stdout, "[I] %s", str);
  fflush(stdout);
}

void logwarn(char * str)
{
  if(loglevel > L_WARN) return;
  fprintf(stderr, "[W] %s", str);
}

void logerr(char * str)
{
  if(loglevel > L_ERR) return;
  fprintf(stderr, "[E] %s", str);
}
