#include "mpf.h"
#include "sys/time.h"
#include "time.h"   /* uses clock for measuring runtime */

/*------------------------ functions used for timing ------------------------ */
double mpf_time
(
  struct timespec start,
  struct timespec finish
)
{
  return (finish.tv_sec - start.tv_sec)
    + (finish.tv_nsec - start.tv_nsec)/1000000000.0;
}
 
void mpf_time_reset
(
  struct timespec *start,
  struct timespec *finish
)
{
  start->tv_sec = 0.0;
  start->tv_nsec = 0.0;
  finish->tv_sec = 0.0;
  finish->tv_nsec = 0.0;
}
 
int mpf_elapsed(double *sec)
{
   //extern int gettimeofday();
   struct timeval t;
   struct timezone tz;
   int stat;
   stat = gettimeofday(&t, &tz);
   *sec = (double)(t.tv_sec + t.tv_usec/1000000.0);
   return(stat);
}
