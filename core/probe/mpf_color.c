#include "mpf.h"

pthread_barrier_t barrier;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex;

/* --------------------------- graph coloring ------------------------------- */

void mpf_color
(
  MPF_Probe *probe
)
{
  MPF_Int z = 0;
  MPF_Int max_coloring = 0;
  MPF_Int source_coloring = 0;
  MPF_Int source = 0;
  MPF_Int destination = 0;
  MPF_Int neighbor = 0;
  MPF_Int previous = 0;

  MPF_Sparse *P = &probe->P;

  MPF_LinkedList *list = mpf_linked_list_create(P->m);
  mpf_matrix_i_set(MPF_COL_MAJOR, P->m, 1, probe->colorings_array, P->m, -1);

  for (source = 0; source < P->m; ++source)
  {
    list->end_internal = -1;
    source_coloring = 0;
    list->start = 0;
    list->end = 0;
    list->id[list->start] = -1;
    list->n_entries = 0;

    for (neighbor = 0; neighbor < P->mem.csr.re[source]-P->mem.csr.rs[source]; ++neighbor)
    {
      destination = P->mem.csr.cols[P->mem.csr.rs[source]+neighbor];

      // @TODO: this should be replaced with a tree data structure.
      // @NOTE: not the bottleneck, so using the linked list is OK for now.

      previous = -1;
      z = list->start;
      while ((z != -1) && (list->id[z] < probe->colorings_array[destination]))
      {
        previous = z;
        z = list->next[z];
      }

      list->end_internal += 1;
      list->n_entries += 1;
      /*
                  z
          A---B---C
      */

      if (list->n_entries == 1)
      {
        list->start = list->end_internal;
        list->end = list->start;
        list->next[list->end_internal] = -1;
      }
      else if ((z == list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = list->start;
        list->start = list->end_internal;
      }
      else if ((z != list->start) && (previous == list->end))
      {
        list->next[previous] = list->end_internal;
        list->next[list->end_internal] = -1;
        list->end = list->end_internal;
      }
      else if ((z != list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = z;
        list->next[previous] = list->end_internal;
      }
      // update data
      list->id[list->end_internal] = probe->colorings_array[destination];
    }

    source_coloring = 0;
    /* parse through list to find the minimum available color to set the
       new node */
    for (z = list->start; z != -1; z = list->next[z])
    {
      if (source_coloring == list->id[z])
      {
        source_coloring += 1;
      }
      else if (source_coloring < list->id[z])
      {
        break;
      }
    }

    probe->colorings_array[source] = source_coloring;
    if (source_coloring > max_coloring)
    {
      max_coloring = source_coloring;
    }
  }
  probe->n_colors = max_coloring+1;
  mpf_linked_list_destroy(list);
}

void mpf_color_decoupled
(
  MPF_Sparse* P,
  MPF_Int* colorings_array,
  MPF_Int* ncolors
)
{
  MPF_Int z = 0;
  MPF_Int max_coloring = 0;
  MPF_Int source_coloring = 0;
  MPF_Int source = 0;
  MPF_Int destination = 0;
  MPF_Int neighbor = 0;
  MPF_Int previous = 0;

  MPF_LinkedList *list = mpf_linked_list_create(P->m);
  mpf_matrix_i_set(MPF_COL_MAJOR, P->m, 1, colorings_array, P->m, -1);

  for (source = 0; source < P->m; ++source)
  {
    list->end_internal = -1;
    source_coloring = 0;
    list->start = 0;
    list->end = 0;
    list->id[list->start] = -1;
    list->n_entries = 0;

    for (neighbor = 0; neighbor < P->mem.csr.re[source]-P->mem.csr.rs[source]; ++neighbor)
    {
      destination = P->mem.csr.cols[P->mem.csr.rs[source]+neighbor];

      // @TODO: this should be replaced with a tree data structure.
      // @NOTE: not the bottleneck, so using the linked list is OK for now.

      previous = -1;
      z = list->start;
      while ((z != -1) && (list->id[z] < colorings_array[destination]))
      {
        previous = z;
        z = list->next[z];
      }

      list->end_internal += 1;
      list->n_entries += 1;
      /*
                  z
          A---B---C
      */

      if (list->n_entries == 1)
      {
        list->start = list->end_internal;
        list->end = list->start;
        list->next[list->end_internal] = -1;
      }
      else if ((z == list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = list->start;
        list->start = list->end_internal;
      }
      else if ((z != list->start) && (previous == list->end))
      {
        list->next[previous] = list->end_internal;
        list->next[list->end_internal] = -1;
        list->end = list->end_internal;
      }
      else if ((z != list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = z;
        list->next[previous] = list->end_internal;
      }
      // update data
      list->id[list->end_internal] = colorings_array[destination];
    }

    source_coloring = 0;
    /* parse through list to find the minimum available color to set the
       new node */
    for (z = list->start; z != -1; z = list->next[z])
    {
      if (source_coloring == list->id[z])
      {
        source_coloring += 1;
      }
      else if (source_coloring < list->id[z])
      {
        break;
      }
    }

    colorings_array[source] = source_coloring;
    if (source_coloring > max_coloring)
    {
      max_coloring = source_coloring;
    }
  }
  *ncolors = max_coloring+1;
  mpf_linked_list_destroy(list);
}

void mpf_color_partial
(
  MPF_Probe *probe,
  MPF_Int current_row,
  MPF_Int current_blk,
  MPF_LinkedList *list
)
{
  MPF_Int z = 0;
  MPF_Int max_coloring = probe->n_colors-1;
  MPF_Int source_coloring = 0;
  MPF_Int source = 0;
  MPF_Int destination = 0;
  MPF_Int neighbor = 0;
  MPF_Int previous = 0;

  //for (source = current_row; source < current_row+current_blk; ++source)
  for (source = 0; source < current_blk; ++source)
  {
    list->end_internal = -1;
    source_coloring = 0;
    list->start = 0;
    list->end = 0;
    list->id[list->start] = -1;
    list->n_entries = 0;
    source_coloring = 0;

    MPF_Int row_start = probe->P.mem.csr.rs[source];
    MPF_Int row_end = probe->P.mem.csr.re[source];

    for (neighbor = 0; neighbor < row_end-row_start; ++neighbor)
    {
      MPF_Int row_index = probe->P.mem.csr.rs[source]+neighbor;
      destination = probe->P.mem.csr.cols[row_index];

      // @TODO: this should be replaced with a tree data structure.
      // @NOTE: not the bottleneck so using the linked list is OK for now.

      previous = -1;
      z = list->start;
      while ((z != -1) && (list->id[z] < probe->colorings_array[destination]))
      {
        previous = z;
        z = list->next[z];
      }

      list->end_internal += 1;
      list->n_entries += 1;
      /*
                  z
          A---B---C
      */

      if (list->n_entries == 1)
      {
        list->start = list->end_internal;
        list->end = list->start;
        list->next[list->end_internal] = -1;
      }
      else if ((z == list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = list->start;
        list->start = list->end_internal;
      }
      else if ((z != list->start) && (previous == list->end))
      {
        list->next[previous] = list->end_internal;
        list->next[list->end_internal] = -1;
        list->end = list->end_internal;
      }
      else if ((z != list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = z;
        list->next[previous] = list->end_internal;
      }
      /* update data */
      list->id[list->end_internal] = probe->colorings_array[destination];
    }

    source_coloring = 0;
    /* parse through list to find the minimum available color to set the
       new node */
    for (z = list->start; z != -1; z = list->next[z])
    {
      if (source_coloring == list->id[z])
      {
        source_coloring += 1;
      }
      else if (source_coloring < list->id[z])
      {
        break;
      }
    }

    probe->colorings_array[source+current_row] = source_coloring;
    if (source_coloring > max_coloring)
    {
      max_coloring = source_coloring;
    }

    if (max_coloring+1 > probe->n_colors)
    {
      probe->n_colors += 1;
    }
  }
}

void mpf_color_unordered
(
  MPF_Probe *probe,
  MPF_Int max_coloring_init,
  MPF_Int n_V,
  MPF_Int *V
)
{
  MPF_Int i = 0;
  MPF_Int z = 0;
  MPF_Int source_coloring = 0;
  MPF_Int source = 0;
  MPF_Int destination = 0;
  MPF_Int neighbor = 0;
  MPF_Int previous = 0;
  MPF_Int max_coloring = 0;

  MPF_LinkedList *list = mpf_linked_list_create(probe->P.m);

  /* @NOTE initialization is not used for augmenetaion purposes, but initial */
  /*  values are used                                                        */

  for (i = 0; i < n_V; ++i)
  {
    source = V[i];
    list->end_internal = -1;
    source_coloring = 0;
    list->start = 0;
    list->end = 0;
    list->id[list->start] = -1;
    list->n_entries = 0;
    //printf("n_V: %d, source: %d\n", n_V, source);
    MPF_Int row_start = probe->P.mem.csr.rs[source];
    MPF_Int row_end = probe->P.mem.csr.re[source];

    for (neighbor = 0; neighbor < row_end - row_start; ++neighbor)
    {
      MPF_Int row_index = probe->P.mem.csr.rs[source]+neighbor;
      destination = probe->P.mem.csr.cols[row_index];

      /* @TODO: this should be replaced with a tree data structure.        */
      /* @NOTE: not the bottleneck so using the linked list is OK for now. */

      previous = -1;
      z = list->start;
      while
      (
        (z != -1) &&
        (list->id[z] < probe->colorings_array[destination])
      )
      {
        previous = z;
        z = list->next[z];
      }

      list->end_internal += 1;
      list->n_entries += 1;
      /*
                  z
          A---B---C
      */

      if (list->n_entries == 1)
      {
        list->start = list->end_internal;
        list->end = list->start;
        list->next[list->end_internal] = -1;
      }
      else if ((z == list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = list->start;
        list->start = list->end_internal;
      }
      else if ((z != list->start) && (previous == list->end))
      {
        list->next[previous] = list->end_internal;
        list->next[list->end_internal] = -1;
        list->end = list->end_internal;
      }
      else if ((z != list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = z;
        list->next[previous] = list->end_internal;
      }
      /* update data */
      list->id[list->end_internal] = probe->colorings_array[destination];
    }

    source_coloring = max_coloring_init+1;
    /* parse list to find the minimum available color to set the */
    /* new node                                                  */
    for (z = list->start; z != -1; z = list->next[z])
    {
      if (source_coloring == list->id[z])
      {
        source_coloring += 1;
      }
      else if (source_coloring < list->id[z])
      {
        break;
      }
    }
    probe->colorings_array[source] = source_coloring;
    if (source_coloring > max_coloring)
    {
      max_coloring = source_coloring;
    }
  }

  probe->n_colors = max_coloring+1;
  mpf_linked_list_destroy(list);
}

void mpf_pthread_color
(
  MPF_Probe *probe
)
{
  MPF_Int max_coloring = 0;

  MPF_PthreadContext_Coloring *t_probe = (MPF_PthreadContext_Coloring*)mpf_malloc(sizeof(MPF_PthreadContext_Coloring)*probe->n_threads);
  MPF_PthreadContextShared_Coloring *t_shared = (MPF_PthreadContextShared_Coloring *) mpf_malloc(sizeof *t_shared);
  t_probe[0].shared_array = (MPF_Int *) mpf_malloc(sizeof(MPF_Int)*probe->P.m);
  t_probe[0].shared_P = &probe->P;
  t_probe[0].thread_id = 0;
  t_probe[0].m = probe->P.m;
  t_probe[0].n_threads = probe->n_threads;
  t_probe[0].max_coloring = 0;

  for (MPF_Int i = 1; i < probe->n_threads; ++i)
  {
    t_probe[i].thread_id = i;
    t_probe[i].shared_array = t_probe[0].shared_array;
    t_probe[i].m = probe->P.m;
    t_probe[i].n_threads = probe->n_threads;
    t_probe[i].max_coloring = -1;
    t_probe[i].shared_P = t_probe[0].shared_P;
  }

  pthread_barrier_init(&barrier, NULL, probe->n_threads);

  /* initialization */
  for (MPF_Int i = 0; i < probe->n_threads; ++i)
  {
    mpf_linked_list_init(probe->P.m, &t_probe[i].list);
    t_probe[i].shared_colorings = probe->colorings_array;
    t_probe[i].shared_scalars = t_shared;
  }

  /* parallel execution of inner solver */
  t_shared->m_array = 0;
  for (MPF_Int i = 0; i < probe->n_threads; ++i)
  {
    pthread_create
    (
      &t_probe[i].pthread_id,
      NULL,
      mpf_pthread_kernel_color,
      &t_probe[i]
    );
  }

  /* join threads */
  for (MPF_Int i = 0; i < probe->n_threads; ++i)
  {
    pthread_join(t_probe[i].pthread_id, NULL);
  }

  for (MPF_Int i = 0; i < probe->n_threads; ++i)
  {
    if (t_probe[i].max_coloring > max_coloring)
    {
      max_coloring = t_probe[i].max_coloring;
    }
  }

  mpf_color_unordered(probe, max_coloring,
    t_probe[0].shared_scalars->m_array, t_probe[0].shared_array);

  for (MPF_Int i = 0; i < probe->n_threads; ++i)
  {
    mpf_linked_list_free(&t_probe[i].list);
  }

  mpf_free(t_probe[0].shared_array);
  mpf_free(t_shared);
  mpf_free(t_probe);
}

void *mpf_pthread_kernel_color
(
  void *input_packed
)
{
  MPF_PthreadContext_Coloring *t_probe = (MPF_PthreadContext_Coloring*)input_packed;
  MPF_LinkedList *list = &t_probe->list;
  MPF_Sparse *P = t_probe->shared_P;
  MPF_Int m = t_probe->m;
  MPF_Int source = 0;
  MPF_Int dest = 0;
  MPF_Int neighbor = 0;
  //MPF_Int job_id = 0;
  MPF_Int n_threads = t_probe->n_threads;
  MPF_Int max_coloring = 0;
  MPF_Int source_coloring = 0;
  MPF_Int z = 0;
  MPF_Int previous = -1;

  MPF_Int blk_thread = (m+n_threads-1)/n_threads;
  MPF_Int len_colorings = (1-(t_probe->thread_id+1)/t_probe->n_threads)*blk_thread + ((t_probe->thread_id+1)/t_probe->n_threads)*(m-(n_threads-1)*blk_thread);

  MPF_Int start = blk_thread*t_probe->thread_id;
  //MPF_Int end = blk_thread*(t_probe->thread_id+1);
  MPF_Int end = (1-(t_probe->thread_id+1)/t_probe->n_threads)*blk_thread*(t_probe->thread_id+1) + ((t_probe->thread_id+1)/t_probe->n_threads)*m;

  /* initialization */
  mpf_matrix_i_set(MPF_COL_MAJOR, len_colorings, 1,
    &t_probe->shared_colorings[blk_thread*t_probe->thread_id], len_colorings, -1);

  /* phase 1: local coloring */
  for (MPF_Int source = start; source < end; ++source)
  {
    source_coloring = 0;
    list->end_internal = -1;
    list->start = 0;
    list->end = 0;
    list->id[list->start] = -1;
    list->n_entries = 0;

    MPF_Int row_start = P->mem.csr.rs[source];
    MPF_Int row_end = P->mem.csr.re[source];

    for (MPF_Int neighbor = 0; neighbor < row_end - row_start; ++neighbor)
    {
      MPF_Int row_index = P->mem.csr.rs[source]+neighbor;
      dest = P->mem.csr.cols[row_index];

      /* @TODO: this should be replaced with a tree data structure.        */
      /* @NOTE: not the bottleneck so using the linked list is OK for now. */

      previous = -1;
      z = list->start;
      while ((z != -1) && (list->id[z] < t_probe->shared_colorings[dest]))
      {
        previous = z;
        z = list->next[z];
      }

      list->end_internal += 1;
      list->n_entries += 1;
      /*
                  z
          A---B---C
      */

      if (list->n_entries == 1)
      {
        list->start = list->end_internal;
        list->end = list->start;
        list->next[list->end_internal] = -1;
      }
      else if ((z == list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = list->start;
        list->start = list->end_internal;
      }
      else if ((z != list->start) && (previous == list->end))
      {
        list->next[previous] = list->end_internal;
        list->next[list->end_internal] = -1;
        list->end = list->end_internal;
      }
      else if ((z != list->start) && (previous != list->end))
      {
        list->next[list->end_internal] = z;
        list->next[previous] = list->end_internal;
      }

      /* updates data */
      list->id[list->end_internal] = t_probe->shared_colorings[dest];
    }

    /* parse list to find the minimum available color to set the */
    /* new node                                                  */
    source_coloring = 0;
    for (z = list->start; z != -1; z = list->next[z])
    {
      if (source_coloring == list->id[z])
      {
        source_coloring += 1;
      }
      else if (source_coloring < list->id[z])
      {
        break;
      }
    }

    t_probe->shared_colorings[source] = source_coloring;
    if (source_coloring > max_coloring)
    {
      max_coloring = source_coloring;
    }
  }
  pthread_barrier_wait(&barrier);

  /* phase 2: finding the nodes to be refined in phase 3 */
  for (source = start; source < end; ++source)
  {
    for (neighbor = 0; neighbor < P->mem.csr.re[source] - P->mem.csr.rs[source]; ++neighbor)
    {
      MPF_Int row_index = P->mem.csr.rs[source]+neighbor;
      dest = P->mem.csr.cols[row_index];
      if
      (
        (t_probe->shared_colorings[source] == t_probe->shared_colorings[dest]) &&
        (source != dest) &&
        (dest >= start) && (dest < end)
      )
      {
        pthread_mutex_lock(&mutex);
        t_probe->shared_array[t_probe->shared_scalars->m_array] = mpf_i_min(source, dest);
        t_probe->shared_scalars->m_array += 1;
        pthread_mutex_unlock(&mutex);
      }
    }
  }

  t_probe->max_coloring = max_coloring;
  return NULL;
}
