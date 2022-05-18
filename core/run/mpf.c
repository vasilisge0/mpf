#include "mpf.h"

void mpf
(
  int argc,
  char **argv
)
{
  /* initializes mpf context */
  MPF_ContextHandle mpf_handle;
  mpf_context_create(&mpf_handle, argc, argv);

  /* runs the multilevel probing method */
  mpf_run(mpf_handle);

  /* displays meta data from the execution step */
  mpf_printout(mpf_handle);

  /* writes approximated fA in file (output) */
  mpf_context_write_fA(mpf_handle);

  /* saves metadata from the execution of mpf */
  mpf_context_write(mpf_handle);

  /* destroys mpf handle */
  mpf_context_destroy(mpf_handle);
}
