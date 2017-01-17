#ifndef _LOG_GUARD
#define _LOG_GUARD

typedef enum {L_DEBUG, L_INFO, L_WARN, L_ERR} loglevel_type;

void logdebug(char * str);
void loginfo(char * str);
void logwarn(char * str);
void logerr(char * str);

#endif
