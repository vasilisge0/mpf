#include "mpf.h"

/*------------------------- fibonacci heap functions -------------------------*/

/*============================================================================*/
/* Allocates memory for internal arrays of fibonacci heap.                    */
/*============================================================================*/
void mpf_heap_min_fibonacci_internal_alloc
(
  MPF_HeapMin_Fibonacci *T
)
{
  //T->degree_markings_length = ceil(log(sqrt(5)*(T->max_num_nodes+1))/log(PHI))-2;
  //T->degree_markings = mpf_malloc(sizeof(MPF_Int)*T->degree_markings_length);

  T->key      = (MPF_Int*)mpf_malloc(sizeof(MPF_Int)*(T->m_max)*8);
  T->deg_mark = &T->key[T->m_max];
  T->deg      = &T->deg_mark[T->m_max];
  T->mark     = &T->deg[T->m_max];
  T->parent   = &T->mark[T->m_max];
  T->previous = &T->parent[T->m_max];
  T->next     = &T->previous[T->m_max];
  T->child    = &T->next[T->m_max];

  T->map1 = NULL;
  T->map2 = NULL;

  mpf_matrix_i_set(MPF_COL_MAJOR, T->m_max*8, 1, T->key, T->m_max*8, -1);
}

void mpf_heap_min_fibonacci_reset
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m
)
{
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, T->key, m, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, T->deg_mark, m, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, T->deg, m, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, T->mark, m, MPF_HEAP_NULL);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, T->parent, m, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, T->previous, m, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, T->next, m, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, m, 1, T->child, m, -1);
}

/*============================================================================*/
/* Initializes the fibonacci heap.                                            */
/*============================================================================*/
void mpf_heap_min_fibonacci_init
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m_max,
  MPF_Int inc
)
{
  T->n_roots = 0;
  T->m = 0;
  T->m_max = m_max;
  T->deg_mark_length = ceil(log(sqrt(5)*(T->m_max+1))/log(PHI))-2;
  T->root_first = 0;
  T->root_last = 0;
  T->root_new = 0;
  T->min_index = 0;
  T->mem_increment = inc;
}

/*============================================================================*/
/* Reallocs memory for the fibonacci heap.                                 */
/*============================================================================*/
void mpf_heap_min_fibonacci_internal_realloc
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int inc
)
{
  MPF_Int m_max_new = T->m_max + inc;

  /* reallocation */
  T->key = (MPF_Int*)mkl_realloc(T->key, sizeof(MPF_Int)*(T->m_max+inc)*8);

  /* move all data (ERROR HERE) */
  memcpy(&T->key[(T->m_max+inc)*1], T->deg_mark, (sizeof *T->deg_mark)*T->m_max);
  memcpy(&T->key[(T->m_max+inc)*2], T->deg, (sizeof *T->deg)*T->m_max);
  memcpy(&T->key[(T->m_max+inc)*3], T->mark, (sizeof *T->mark)*T->m_max);
  memcpy(&T->key[(T->m_max+inc)*4], T->parent, (sizeof *T->parent)*T->m_max);
  memcpy(&T->key[(T->m_max+inc)*5], T->previous, (sizeof *T->previous)*T->m_max);
  memcpy(&T->key[(T->m_max+inc)*6], T->next, (sizeof *T->next)*T->m_max);
  memcpy(&T->key[(T->m_max+inc)*7], T->child, (sizeof *T->child)*T->m_max);

  T->deg_mark = &T->key[(T->m_max+inc)*1];
  T->deg = &T->key[(T->m_max+inc)*2];
  T->mark = &T->key[(T->m_max+inc)*3];
  T->parent = &T->key[(T->m_max+inc)*4];
  T->previous = &T->key[(T->m_max+inc)*5];
  T->next = &T->key[(T->m_max+inc)*6];
  T->child = &T->key[(T->m_max+inc)*7];

  mpf_matrix_i_set(MPF_COL_MAJOR, inc, 1, &T->key[T->m_max], m_max_new, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, inc, 1, &T->deg_mark[T->m_max], m_max_new, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, inc, 1, &T->deg[T->m_max], m_max_new, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, inc, 1, &T->mark[T->m_max], m_max_new,
    MPF_HEAP_NULL);
  mpf_matrix_i_set(MPF_COL_MAJOR, inc, 1, &T->parent[T->m_max], m_max_new, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, inc, 1, &T->previous[T->m_max], m_max_new, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, inc, 1, &T->next[T->m_max], m_max_new, -1);
  mpf_matrix_i_set(MPF_COL_MAJOR, inc, 1, &T->child[T->m_max], m_max_new, -1);

  T->m_max = m_max_new;
}

/*============================================================================*/
/* Deallocs used memory.                                                      */
/*============================================================================*/
void mpf_heap_min_fibonacci_internal_free
(
  MPF_HeapMin_Fibonacci *T
)
{
  mpf_free(T->key);
}

/*============================================================================*/
/* mpf_heap_min_fibonacci_insert                                               */
/*                                                                            */
/*                        root_new                                            */
/*                            v                                               */
/*    |----------------xxxxxxx----------|                                     */
/*     ^              ^=======                                                */
/* root_first     root_last ^                                                 */
/*                          |                                                 */
/*                       lost memory entries, refragmantation should be called*/
/*                       at some point.                                       */
/*                                                                            */
/*============================================================================*/
void mpf_heap_min_fibonacci_insert
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int new_key
)
{
  /* alloc more memory, nodes are added as root to the end of the root linked
     list, so you should look at root_end */

  //printf("T->root_new: %d,T->m_max: %d\n", T->root_new, T->m_max);
  //if
  //(
  //  (T->root_new == T->m_max) &&
  //  (((double)T->n_null)/(T->m_max) < MPF_HEAP_DEFRAG_TRHES)
  //)
  //{
  //  mpf_heap_min_fibonacci_defragment(T);
  //}
  //else if
  //(
  //  ((T->root_new == T->m_max) &&
  //  ((((double)T->n_null)/T->m_max) >= MPF_HEAP_DEFRAG_TRHES))
  //)
  //{
  //  mpf_heap_min_fibonacci_internal_realloc(T, T->memory_inc);
  //  T->m_max += T->memory_inc;
  //}

  if
  (
    (T->root_new == T->m_max)
  )
  {
    mpf_heap_min_fibonacci_internal_realloc(T, T->mem_increment);
    T->m_max += T->mem_increment;
  }

  T->key[T->root_new] = new_key;  /* sets new key */
  T->deg[T->root_new] = 0;        /* sets new degree to 0 */
  T->mark[T->root_new] = MPF_HEAP_UNMARKED;

  if (T->m > 0)
  {
    /* checks if new_node has the minimum key, and updates it if it does */
    if (T->key[T->root_new] < T->key[T->min_index])
    {
      T->min_index = T->root_new;
    }

    T->previous[T->root_new] = T->root_last;  /*     backward link: root_last <- root_new */
    T->next[T->root_last] = T->root_new;      /*      forward link: root_last -> root_new */
    T->next[T->root_new] = -1;                /* null forward link: root_new ->|| */
    T->root_last = T->root_new;               /* last root node root_new = root_last */
    T->root_new += 1;                         /* update new node */
  }
  else if (T->m == 0)
  {
    T->min_index = 0;
    T->root_first = 0;
    T->root_last = 0;
    T->root_new = 1;
    T->next[0] = -1;
    T->previous[0] = -1;
  }
  T->m += 1;
}

/*============================================================================*/
/* Extracts minimum from the fibonacci heap.                                  */
/*============================================================================*/
MPF_Int mpf_heap_min_fibonacci_extract_min
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int *return_key
)
{
    MPF_Int return_index = -1;

    if (T->m > 0)
    {
      return_index = T->min_index;
      *return_key = T->key[T->min_index];
      mpf_heap_min_fibonacci_delete_min(T);
    }
    else
    {
      *return_key = -1;
    }

    return return_index;
}

/*============================================================================*/
/* Defragments the allocd space by a fibonacci mean_heap.                  */
/*============================================================================*/
void mpf_heap_min_fibonacci_defragment
(
  MPF_HeapMin_Fibonacci *T
)
{
  MPF_Int r = 0;
  MPF_Int rc = 0;
  MPF_Int n_parsed = 0;
  MPF_Int curr = 0;
  MPF_Int i = 0;

  for (r = T->root_first; r != -1; r = T->next[r])
  {
    rc = r;
    while (rc > -1)
    {
      for (i = T->child[rc]; i != -1; i = T->next[i])
      {
        while ((curr != MPF_HEAP_NULL) && (curr < n_parsed))
        {
          curr+=1;
        }

        if ((curr < n_parsed))
        {
          mpf_heap_min_fibonacci_node_move(T, i, curr);
          n_parsed += 1;
        }

        if ((T->child[i] > -1) && (T->previous[T->child[i]] == -1))
        {
          rc = T->child[i];
        }
        else
        {
          rc = -1;
        }
      }
    }
  }
}

/*============================================================================*/
/* Moves node in the heap from source -> dest.                                */
/*============================================================================*/
void mpf_heap_min_fibonacci_node_move
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int source,
  MPF_Int dest
)
{
  T->parent[dest] = T->parent[source];
  T->parent[source] = -1;

  T->child[T->parent[source]] = dest;

  T->next[dest] = T->next[source];
  T->next[source] = -1;

  if (T->previous[source] > -1)
  {
    T->next[T->previous[source]] = dest;
    T->previous[dest] = -1;
  }

  T->deg[dest] = T->deg[source];
  T->deg[source] = -1;

  T->mark[dest] = T->mark[source];
  T->mark[source] = MPF_HEAP_NULL;

  T->key[dest] = T->key[source];
  T->key[source] = -1;

  if (source == T->root_first)
  {
    T->root_first = dest;
  }

  if (source == T->root_last)
  {
    T->root_last = dest;
  }

  if (source == T->min_index)
  {
    T->min_index = source;
  }
}

#define DEBUG_HEAP 0

/*============================================================================*/
/* Deletes the node with the minimum key.                                     */
/*============================================================================*/
void mpf_heap_min_fibonacci_delete_min
(
  MPF_HeapMin_Fibonacci *T
)
{
  MPF_Int i = 0;      /* parses nodes */
  MPF_Int next = -1;  /* for finding next root */

  #if DEBUG_HEAP == 1
    printf("\n     >>>>>>>>>>>> extracting [T->m: %d] <<<<<<<<<<<<\n", T->m);
    printf("\n ");
    mpf_heap_min_fibonacci_view_roots(T);
    printf("\n extracting node: %d\n", T->min_index);
    printf("T->min_index: %d -> T->key[T->min_index]: %d\n",
      T->min_index, T->key[T->min_index]);
    printf(" -> T->deg[%d]: %d\n", T->min_index, T->deg[T->min_index]);
    printf(" -> [root_first: %d, root_last: %d]\n\n",
      T->root_first, T->root_last);
    printf("T->next[T->root_first]: %d\n", T->next[T->root_first]);
  #endif

  /*----------------------*/
  /* removes T->min_index */
  /*----------------------*/
  if (T->m > 1) /* when you have more than 1 entry */
  {
    if (T->deg[T->min_index] > 0) /* T->min_index has children */
    {
      /*----------------------------------*/
      /* resets T->previous[T->min_index] */
      /*----------------------------------*/
      if (T->previous[T->min_index] > -1) /* it has previous node */
      {
        T->next[T->previous[T->min_index]] = T->child[T->min_index];  /* sets the next of the previous */
        #if DEBUG_HEAP == 1
          printf("   T->child[T->min_index]: %d]\n", T->child[T->min_index]);
          printf("T->previous[T->min_index]: %d\n", T->previous[T->min_index]);
        #endif
        T->previous[T->child[T->min_index]] = T->previous[T->min_index]; /* sets previous of child */
      }
      else if (T->previous[T->min_index] == -1) /* if previous is null */
      {
        T->previous[T->child[T->min_index]] = -1; /* sets previous of child to null */
        T->root_first = T->child[T->min_index];   /* sets child as first root */
      }

      {
        MPF_Int last_child = T->child[T->min_index];  /* moves to last child */
        MPF_Int c = 0;
        MPF_Int count = 0;
        #if DEBUG_HEAP == 1
          printf("T->min_index: %d\n", T->min_index);
          printf("  last_child: %d\n", last_child);
          printf("T->next[last_child]: %d\n", T->next[last_child]);
          printf("T->next[T->next[last_child]]: %d\n",
            T->next[T->next[last_child]]);
        #endif
        while (last_child > -1)
        {
          count = 0;
          c = T->child[last_child];
          //printf("c: %d, count: %d, last_child: %d\n", c, count, last_child);
          while (c > -1)
          {
            if (last_child == 3)
            {
              //printf("c: %d\n", c);
            }
            c = T->next[c];
            //printf("count: %d, c: %d\n", count, c);
            count += 1;
          }
          T->deg[last_child] = count;
          last_child = T->next[last_child];
        }
      }

      /*------------------------------*/
      /* resets T->next[T->min_index] */
      /*------------------------------*/
      if (T->next[T->min_index] > -1)
      {
        MPF_Int last_child = T->child[T->min_index];  /* moves to last child */
        while (T->next[last_child] != -1)
        {
          last_child = T->next[last_child];
        }

        T->previous[T->next[T->min_index]] = last_child;  /* previous of next[T->min] */
        T->next[last_child] = T->next[T->min_index];      /* next of last child */
      }
      else if (T->next[T->min_index] == -1)
      {
        MPF_Int last_child = T->child[T->min_index];  /* moves to last child */
        while (T->next[last_child] != -1)
        {
          last_child = T->next[last_child];
        }

        T->next[last_child] = -1;  /* unecessary */
        T->root_last = last_child; /* new last root */
      }

      /*---------------------------------------------------*/
      /* negate pointers from T->min_index that is removed */
      /*---------------------------------------------------*/
      T->child[T->min_index] = -1;    /* no longer a child */
      T->next[T->min_index] = -1;     /* node is removed => has no next */
      T->previous[T->min_index] = -1; /* node is removed => has no previous */
    }
    else if (T->deg[T->min_index] == 0) /* T->min_index root has 0 children */
    {
      /*-----------------------------------------------------*/
      /* T->min_index node is not the first or the last root */
      /*-----------------------------------------------------*/
      if ((T->min_index != T->root_last) && (T->min_index != T->root_first))
      {
        T->next[T->previous[T->min_index]] = T->next[T->min_index];
        T->previous[T->next[T->min_index]] = T->previous[T->min_index];
      }

      /*------------------------------------------*/
      /* T->min_index node is the first root node */
      /*------------------------------------------*/
      if (T->min_index == T->root_first)
      {
        T->previous[T->next[T->min_index]] = -1;
        T->root_first = T->next[T->min_index];
      }

      /*-----------------------------------------*/
      /* T->min_index node is the last root node */
      /*-----------------------------------------*/
      if (T->min_index == T->root_last)
      {
        T->next[T->previous[T->min_index]] = -1;
        T->root_last = T->previous[T->min_index];
      }
    }

    #if DEBUG_HEAP == 1
      printf(" restructuring queue\n");
      printf(" -> [root_first: %d, root_last: %d]\n\n",
        T->root_first, T->root_last);
      printf("\n");
    #endif
    /*---------------------------------------------*/
    /* removes markings of T->degree[T->min_index] */
    /*---------------------------------------------*/
    if
    (
      T->deg_mark[T->deg[T->min_index]] == T->min_index
    )
    {
      T->deg_mark[T->deg[T->min_index]] = -1;
    }
    T->deg[T->min_index] = -1; /* nulls the degree of T->min_index */

    #if DEBUG_HEAP == 1
      printf(" root_first: %d, root_last: %d\n", T->root_first, T->root_last);
      //mpf_heap_min_fibonacci_view_roots(T);
    #endif

    i = T->root_first;  /* start parsing nodes */
    while
    (
      //(i <= T->root_last) &&
      (i > -1) &&
      (T->m > 1)
    )
    {
      next = T->next[i]; /* access next root pointer */
      #if DEBUG_HEAP == 1
        printf(" -> i: %d, root_first: %d, root_last: %d\n",
          i, T->root_first, T->root_last);
        printf(" ----|    T->child[%d]: %d\n", i, T->child[i]);
        printf(" ----|      T->deg[%d]: %d\n", i, T->deg[i]);
      #endif

      if  /* restructure tree */
      (
        (T->deg[i] > -1) && /* i is a root */
        (T->deg_mark[T->deg[i]] > -1) &&  /* there exists another node with same degree */
        (T->deg_mark[T->deg[i]] != i) /* the node is different than i */
      )
      {
        MPF_Int parent = -1;
        MPF_Int child = -1;
        if (T->key[T->deg_mark[T->deg[i]]] <= T->key[i])
        {
          parent = T->deg_mark[T->deg[i]];
          child = i;
        }
        else
        {
          parent = i;
          child = T->deg_mark[T->deg[i]];
        }

        /* merge nodes i and T->degree_markings[T->degree[i]] */

        T->parent[child] = parent; /* set parent of i */

        if (T->previous[child] > -1)
        {
          T->next[T->previous[child]] = T->next[child]; /* fill the gap */
        }

        if (T->next[child] > -1)
        {
          T->previous[T->next[child]] = T->previous[child]; /* fill the gap */
        }

        #if DEBUG_HEAP == 1
          printf("IN:\n");
          mpf_heap_min_fibonacci_view_roots(T);
          printf("root_first: %d, root_last: %d\n",
            T->root_first, T->root_last);
          printf(" ------?   T->parent[%d]: %d\n", i, T->parent[i]);
          printf(" ------?    T->child[%d]: %d\n",
            T->parent[i], T->child[T->parent[i]]);
          printf(" ------?     T->next[%d]: %d\n",
            T->previous[i], T->next[T->previous[i]]);
          printf(" ------? T->previous[%d]: %d\n",
            T->next[i], T->previous[T->next[i]]);
          printf("parent: %d -> %d\n", parent, T->key[parent]);
          printf(" child: %d -> %d\n", child, T->key[child]);
          printf("T->deg_mark[T->deg[i]]: %d -> %d\n",
            T->deg_mark[T->deg[i]], T->key[T->deg_mark[T->deg[i]]]);
          printf("i: %d -> %d\n", i, T->key[i]);
        #endif

        if ((child == T->root_first) && (T->next[child] > -1)) /* sets new root_first */
        {
          T->root_first = T->next[T->root_first];
        }
        else if ((child == T->root_first) && (T->next[child] == -1))
        {
          T->root_first = parent;
        }

        if ((child == T->root_last) && (T->previous[child] > -1)) /* sets new root_last */
        {
          T->root_last = T->previous[T->root_last];
        }
        else if ((child == T->root_last) && (T->previous[child] == -1))
        {
          T->root_last = parent;
        }
        #if DEBUG_HEAP == 1
          mpf_heap_min_fibonacci_view_roots(T);
        #endif

        if (T->child[parent] == -1) /* add i to the child list of its new parent */
        {
          T->child[parent] = child;
          T->next[child] = -1; //@
        }
        else /* there are some other nodes in the children list */
        {
          //original
          //T->next[i] = T->child[T->deg_mark[T->deg[i]]];
          //T->previous[T->child[T->deg_mark[T->deg[i]]]] = i;

          //T->next[T->child[T->deg_mark[T->deg[i]]]] = i;
          MPF_Int child_temp = T->child[parent];
          while (T->next[child_temp] > -1)
          {
            child_temp = T->next[child_temp];
            //printf("child_temp: %d, T->m: %d\n", child_temp, T->m);
          }
          T->next[child_temp] = child;
          T->previous[child] = child_temp;
          T->next[child] = -1;
        }

        /* update degree markings */
        #if DEBUG_HEAP == 1
          mpf_heap_min_fibonacci_view_roots(T);
          printf(">> T->deg_mark[T->deg[parent]+1]: %d\n",
            T->deg_mark[T->deg[parent]+1]);
          printf(">> T->deg[parent]+1: %d\n", T->deg[parent]+1);
          printf("parent: %d\n", parent);
          printf(" child: %d\n", child);
        #endif
        if ((T->deg_mark[T->deg[parent]+1] > -1) &&
            (T->deg_mark[T->deg[parent]+1] != T->parent[child]))
        {
          next = T->parent[child];
        }
        else if (T->deg_mark[T->deg[parent]+1] == -1)
        {
          //T->deg_mark[T->deg[parent]+1] = T->deg_mark[T->deg[parent]];
          T->deg_mark[T->deg[parent]+1] = parent;
        }

        T->deg_mark[T->deg[child]] = -1;
        T->deg[parent] += 1;
        T->deg[child] = -1;

      }
      else if /* updates degree list */
      (
        (T->deg[i] > -1) &&
        (T->deg_mark[T->deg[i]] == -1)
      )
      {
        T->deg_mark[T->deg[i]] = i;
      }
      #if DEBUG_HEAP == 1
        printf("T->next[%d]: %d\n", i, T->next[i]);
        mpf_heap_min_fibonacci_view_roots(T);
        printf("last next: %d\n", next);
      #endif

      /* break when reaches root_last */
      if ((next != -1) || ((next != -1) && (next == T->parent[i])) ||
          ((next != -1) && (T->deg[next] == T->deg[i])))
      {
        i = next;
      }
      else
      {
        #if DEBUG_HEAP == 1
          printf("i: %d\n", i);
          printf("next: %d\n", next);
          printf("T->parent[i]: %d\n", T->parent[i]);
          printf("T->deg_mark[T->deg[i]]: %d\n", T->deg_mark[T->deg[i]]);
          printf("*>Quit\n");
        #endif
        break;
      }
    }

    #if DEBUG_HEAP == 1
      printf("after restructuring\n");
      printf("root_first: %d\n", T->root_first);
      printf("root_last: %d\n", T->root_last);
      printf("T->min_index: %d -> %d\n", T->min_index, T->key[T->min_index]);
      printf("T->next[T->root_first]: %d\n", T->next[T->root_first]);
      mpf_heap_min_fibonacci_view_roots(T);
    #endif

    /*--------------------*/
    /* find new min_index */
    /*--------------------*/
    T->min_index = T->root_first;
    i = T->next[T->root_first];
    MPF_Int min_new = T->root_first;
    T->min_index = min_new;
    while
    (
      (min_new > -1)// &&
      //(i <= T->root_last)
    )
    {
      if (T->key[min_new] < T->key[T->min_index])
      {
        T->min_index = min_new;
      }
      min_new = T->next[min_new];
    }
    T->mark[T->min_index] = MPF_HEAP_UNMARKED;
  }
  else if (T->m == 1)
  {
    T->mark[T->min_index] = MPF_HEAP_UNMARKED;
    T->next[T->min_index] = -1;
    T->previous[T->min_index] = -1;
    T->min_index = -1;
    T->root_new = 0;
    T->root_first = -1;
    T->root_last = -1;
  }
  T->m -= 1;
}

void mpf_heap_min_fibonacci_children_update
(
  MPF_HeapMin_Fibonacci *T
)
{
  /* not used */
}

/*============================================================================*/
/* Displays the roots of the fibonacci min_heap                               */
/*============================================================================*/
void mpf_heap_min_fibonacci_view_roots
(
  MPF_HeapMin_Fibonacci *T
)
{
  printf("roots: ");
  MPF_Int i = T->root_first;
  printf("(");
  if (T->m > 0)
  {
    while (i > -1)
    {
      printf("%d:>%d|%d", i, T->key[i], T->deg[i]);
      i = T->next[i];
      if (i > -1)
      {
        printf(" ");
      }
      else
      {
        break;
      }
    }
  }
  else if (T->m == 0)
  {
    printf("empty");
  }
  printf(")\n");
}

/*============================================================================*/
/* Plots root nodes of the fibonacci heap.                                    */
/*============================================================================*/
void mpf_heap_min_fibonacci_plot_roots
(
  MPF_HeapMin_Fibonacci *T
)
{
  MPF_Int i = T->root_first;
  if (T->m > 0)
  {
    while ((i <= T->root_last) && (i > -1))
    {
      if (T->next[i] > -1)
      {
        i = T->next[i];
      }
      else
      {
        break;
      }
    }
  }
  else if (T->m == 0)
  {
    printf("empty");
  }
}

/*============================================================================*/
/* Decreases the key of a node in the fibonacci heap.                         */
/*============================================================================*/
void mpf_heap_min_fibonacci_decrease
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int i,
  MPF_Int new_key
)
{
  MPF_Int j = 0;
  if (T->key[i] < new_key)
  {
    T->key[i] = new_key;
    T->mark[i] = MPF_HEAP_UNMARKED;
    if (T->key[T->child[i]] < T->key[i])
    {
      j = T->parent[i];
      T->parent[i] = -1;
      T->next[T->root_last] = i;
      T->root_last = i;
      while (T->mark[j] == MPF_HEAP_MARKED)
      {
        T->mark[j]= MPF_HEAP_UNMARKED;
        T->next[T->root_last] = j;
        T->parent[j] = -1;
        T->previous[j] = T->root_last;
        T->root_last = j;
        j = T->parent[i];
      }
      if (T->parent[j] != -1)
      {
        T->mark[j] = MPF_HEAP_MARKED;
      }
    }
  }
}

/*============================================================================*/
/* heapsort                                                                   */
/* --------                                                                   */
/* sorts entries in increasing order                                          */
/*============================================================================*/
void mpf_i_heapsort
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m,
  MPF_Int *v
)
{
  MPF_Int i = 0;

  /* construct heap */
  T->m = 0;
  for (i = 0; i < m; ++i)
  {
    mpf_heap_min_fibonacci_insert(T, v[i]);
  }

  //mpf_matrix_i_announce(T->key, m, 1, m, "T->key");

  /* extract every entry of T in increasing order */
  for (i = 0; i < m; ++i)
  {
    mpf_heap_min_fibonacci_extract_min(T, &v[i]);
  }

  /* resets heap */
  mpf_heap_min_fibonacci_reset(T, m);
}

/*============================================================================*/
/* Sorts ids of the nodes that belong on a fibonacci heap.                    */
/*============================================================================*/
void mpf_d_id_heapsort
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m,
  MPF_Int *v,
  double *v_data,
  double *buffer
)
{
  MPF_Int index = 0;

  /* construct heap */
  T->m = 0;
  for (MPF_Int i = 0; i < m; ++i)
  {
    mpf_heap_min_fibonacci_insert(T, v[i]);
  }

  //mpf_matrix_i_announce(T->key, m, 1, m, "T->key");
  memcpy(buffer, v_data, (sizeof *v_data)*m);

  /* extract every entry of T in increasing order */
  for (MPF_Int i = 0; i < m; ++i)
  {
    index = mpf_heap_min_fibonacci_extract_min(T, &v[i]);
    v_data[i] = buffer[index];
  }

  /* resets heap */
  mpf_heap_min_fibonacci_reset(T, m);
}

void mpf_z_id_heapsort
(
  MPF_HeapMin_Fibonacci *T,
  MPF_Int m,
  MPF_Int *v,
  MPF_ComplexDouble *v_data,
  MPF_ComplexDouble *buffer
)
{
  MPF_Int index = 0;

  /* construct heap */
  T->m = 0;
  for (MPF_Int i = 0; i < m; ++i)
  {
    mpf_heap_min_fibonacci_insert(T, v[i]);
  }

  //mpf_matrix_i_announce(T->key, m, 1, m, "T->key");
  memcpy(buffer, v_data, (sizeof *v_data)*m);

  /* extract every entry of T in increasing order */
  for (MPF_Int i = 0; i < m; ++i)
  {
    index = mpf_heap_min_fibonacci_extract_min(T, &v[i]);
    v_data[i] = buffer[index];
  }

  /* resets heap */
  mpf_heap_min_fibonacci_reset(T, m);
}
