#include "mpf.h"

MPF_LinkedList *mpf_linked_list_create(MPF_Int n_entries)
{
  /* allocate */
  MPF_LinkedList *list = (MPF_LinkedList*)mpf_malloc(sizeof *list);
  list->id = (MPF_Int*)mpf_malloc((sizeof list->id)*n_entries);
  list->next = (MPF_Int*)mpf_malloc((sizeof list->next)*n_entries);

  /* initialize id/next to -1 */
  mpf_matrix_i_set(MPF_COL_MAJOR, n_entries, 1, list->id, n_entries, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, n_entries, 1, list->next, n_entries, -1);

  /* initialize metadata */
  list->max_n_entries = n_entries;
  list->end_internal = -1;
  list->start = 0;
  list->end = 0;
  list->id[list->start] = -1;
  list->n_entries = 0;
  return list;
}

void mpf_linked_list_destroy
(
  MPF_LinkedList *list
)
{
  list->max_n_entries = 0;
  list->n_entries = 0;
  list->start = 0;
  list->end = 0;
  list->end_internal = 0;
  mpf_free(list->id);
  mpf_free(list->next);
  //mpf_free(list->previous);
  mpf_free(list);
}

void mpf_linked_list_init
(
  MPF_Int n_entries,
  MPF_LinkedList *list
)
{
  /* allocate */
  list->id = (MPF_Int*)mpf_malloc((sizeof list->id)*n_entries);
  list->next = (MPF_Int*)mpf_malloc((sizeof list->next)*n_entries);

  /* initialize id/next to -1 */
  mpf_matrix_i_set(MPF_COL_MAJOR, n_entries, 1, list->id, n_entries, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, n_entries, 1, list->next, n_entries, -1);

  /* initialize metadata */
  list->max_n_entries = n_entries;
  list->n_entries = 0;
  list->start = 0;
  list->end = 0;
  list->end_internal = 0;
}

void mpf_linked_list_free
(
  MPF_LinkedList *list
)
{
  list->max_n_entries = 0;
  list->n_entries = 0;
  list->start = 0;
  list->end = 0;
  list->end_internal = 0;
  mpf_free(list->id);
  mpf_free(list->next);
}
