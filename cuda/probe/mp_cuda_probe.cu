#include "mp.h"
#include "stdio.h"
#include "cusparse_v2.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include "mp_cuda_probing.h"

//extern "C"

__host__ void mp_cuda_pattern_heap_allocate
(
  MPPatternHeap_Cuda *T
)
{
  int i = 0;
  cudaMalloc((void**)&T->d_heap_array, (sizeof *T->d_heap_array)*T->m);
  cudaMalloc((void**)&T->d_heap_array[0].key, sizeof(MPInt)*T->m*T->m_internal_max*8);
  T->d_heap_array[0].deg_mark = &T->d_heap_array[0].key[T->m_internal_max];
  T->d_heap_array[0].deg      = &T->d_heap_array[0].deg_mark[T->m_internal_max];
  T->d_heap_array[0].mark     = &T->d_heap_array[0].deg[T->m_internal_max];
  T->d_heap_array[0].parent   = &T->d_heap_array[0].mark[T->m_internal_max];
  T->d_heap_array[0].previous = &T->d_heap_array[0].parent[T->m_internal_max];
  T->d_heap_array[0].next     = &T->d_heap_array[0].previous[T->m_internal_max];
  T->d_heap_array[0].child    = &T->d_heap_array[0].next[T->m_internal_max];
  for (i = 1; i < T->m; ++i)
  {
    T->d_heap_array[i].key = &T->d_heap_array[i-1].key[8*T->m_internal_max];
    T->d_heap_array[i].deg_mark = &T->d_heap_array[i].key[T->m_internal_max];
    T->d_heap_array[i].deg      = &T->d_heap_array[i].deg_mark[T->m_internal_max];
    T->d_heap_array[i].mark     = &T->d_heap_array[i].deg[T->m_internal_max];
    T->d_heap_array[i].parent   = &T->d_heap_array[i].mark[T->m_internal_max];
    T->d_heap_array[i].previous = &T->d_heap_array[i].parent[T->m_internal_max];
    T->d_heap_array[i].next     = &T->d_heap_array[i].previous[T->m_internal_max];
    T->d_heap_array[i].child    = &T->d_heap_array[i].next[T->m_internal_max];
  }
}

__host__ void mp_heap_min_cuda_fibonacci_internal_allocate
(
  MPHeapMin_Fibonacci *T
)
{
  //T->degree_markings_length = ceil(log(sqrt(5)*(T->max_num_nodes+1))/log(PHI))-2;
  //T->degree_markings = mp_malloc(sizeof(MPInt)*T->degree_markings_length);

  cudaMalloc((void**)&T->key, sizeof(MPInt)*(T->m_max)*8);
  //T->key      = mp_malloc(sizeof(MPInt)*(T->m_max)*8);
  T->deg_mark = &T->key[T->m_max];
  T->deg      = &T->deg_mark[T->m_max];
  T->mark     = &T->deg[T->m_max];
  T->parent   = &T->mark[T->m_max];
  T->previous = &T->parent[T->m_max];
  T->next     = &T->previous[T->m_max];
  T->child    = &T->next[T->m_max];

  //T->map1 = NULL;
  //T->map2 = NULL;

  //mp_matrix_i_set(MP_COL_MAJOR, T->m_max*8, 1, T->key, T->m_max*8, -1);
}

__device__ void mp_heap_min_cuda_fibonacci_reset
(
  MPHeapMin_Fibonacci *T,
  MPInt m
)
{
  //mp_matrix_i_set(MP_COL_MAJOR, m, 1, T->key, m, -1);
  //mp_matrix_i_set(MP_COL_MAJOR, m, 1, T->deg_mark, m, -1);
  //mp_matrix_i_set(MP_COL_MAJOR, m, 1, T->deg, m, -1);
  //mp_matrix_i_set(MP_COL_MAJOR, m, 1, T->mark, m, MP_HEAP_NULL);
  //mp_matrix_i_set(MP_COL_MAJOR, m, 1, T->parent, m, -1);
  //mp_matrix_i_set(MP_COL_MAJOR, m, 1, T->previous, m, -1);
  //mp_matrix_i_set(MP_COL_MAJOR, m, 1, T->next, m, -1);
  //mp_matrix_i_set(MP_COL_MAJOR, m, 1, T->child, m, -1);
}

__device__ void mp_heap_min_cuda_fibonacci_init
(
  MPHeapMin_Fibonacci *T,
  MPInt m_max,
  MPInt inc
)
{
  T->n_roots = 0;
  T->m = 0;
  T->m_max = m_max;
  T->deg_mark_length = ceil(log2(sqrt(5.0)*(T->m_max+1))/log2(PHI))-2;
  T->root_first = 0;
  T->root_last = 0;
  T->root_new = 0;
  T->min_index = 0;
  T->memory_inc = inc;
}

//void mp_heap_min_cuda_fibonacci_internal_reallocate
//(
//  MPHeapMin_Fibonacci *T,
//  MPInt inc
//)
//{
//  MPInt m_max_new = T->m_max + inc;
//
//  /* reallocation */
//  T->key = mkl_realloc(T->key, sizeof(MPInt)*(T->m_max+inc)*8);
//
//  /* move all data (ERROR HERE) */
//  memcpy(&T->key[(T->m_max+inc)*1], T->deg_mark, (sizeof *T->deg_mark)*T->m_max);
//  memcpy(&T->key[(T->m_max+inc)*2], T->deg, (sizeof *T->deg)*T->m_max);
//  memcpy(&T->key[(T->m_max+inc)*3], T->mark, (sizeof *T->mark)*T->m_max);
//  memcpy(&T->key[(T->m_max+inc)*4], T->parent, (sizeof *T->parent)*T->m_max);
//  memcpy(&T->key[(T->m_max+inc)*5], T->previous, (sizeof *T->previous)*T->m_max);
//  memcpy(&T->key[(T->m_max+inc)*6], T->next, (sizeof *T->next)*T->m_max);
//  memcpy(&T->key[(T->m_max+inc)*7], T->child, (sizeof *T->child)*T->m_max);
//
//  T->deg_mark = &T->key[(T->m_max+inc)*1];
//  T->deg = &T->key[(T->m_max+inc)*2];
//  T->mark = &T->key[(T->m_max+inc)*3];
//  T->parent = &T->key[(T->m_max+inc)*4];
//  T->previous = &T->key[(T->m_max+inc)*5];
//  T->next = &T->key[(T->m_max+inc)*6];
//  T->child = &T->key[(T->m_max+inc)*7];
//
//  mp_matrix_i_set(MP_COL_MAJOR, inc, 1, &T->key[T->m_max], m_max_new, -1);
//  mp_matrix_i_set(MP_COL_MAJOR, inc, 1, &T->deg_mark[T->m_max], m_max_new, -1);
//  mp_matrix_i_set(MP_COL_MAJOR, inc, 1, &T->deg[T->m_max], m_max_new, -1);
//  mp_matrix_i_set(MP_COL_MAJOR, inc, 1, &T->mark[T->m_max], m_max_new, MP_HEAP_NULL);
//  mp_matrix_i_set(MP_COL_MAJOR, inc, 1, &T->parent[T->m_max], m_max_new, -1);
//  mp_matrix_i_set(MP_COL_MAJOR, inc, 1, &T->previous[T->m_max], m_max_new, -1);
//  mp_matrix_i_set(MP_COL_MAJOR, inc, 1, &T->next[T->m_max], m_max_new, -1);
//  mp_matrix_i_set(MP_COL_MAJOR, inc, 1, &T->child[T->m_max], m_max_new, -1);
//
//  T->m_max = m_max_new;
//}

__host__ void mp_heap_min_cuda_fibonacci_internal_free
(
  MPHeapMin_Fibonacci *T
)
{
  cudaFree(T->key);
}

/*==============================================================================

                        root_new
                            v
    |----------------xxxxxxx----------|
     ^              ^=======
 root_first     root_last ^
                          |
                       lost memory entries, refragmantation should be called
                       at some point.

==============================================================================*/
__device__ void mp_heap_min_cuda_fibonacci_insert
(
  MPHeapMin_Fibonacci *T,
  MPInt new_key
)
{
  /* allocate more memory, nodes are added as root to the end of the root linked
     list, so you should look at root_end */

  //printf("T->root_new: %d,T->m_max: %d\n", T->root_new, T->m_max);
  //if
  //(
  //  (T->root_new == T->m_max) &&
  //  (((double)T->n_null)/(T->m_max) < MP_HEAP_DEFRAG_TRHES)
  //)
  //{
  //  mp_heap_min_fibonacci_defragment(T);
  //}
  //else if
  //(
  //  ((T->root_new == T->m_max) &&
  //  ((((double)T->n_null)/T->m_max) >= MP_HEAP_DEFRAG_TRHES))
  //)
  //{
  //  mp_heap_min_fibonacci_internal_reallocate(T, T->memory_inc);
  //  T->m_max += T->memory_inc;
  //}

  if
  (
    (T->root_new == T->m_max)
  )
  {
    //mp_heap_min_fibonacci_internal_reallocate(T, T->memory_inc);
    //T->m_max += T->memory_inc;
    printf("Error at:>mp_heap_min_cuda_fibonacci_insert... Not enough device \
            memory\n");
    return;
  }

  T->key[T->root_new] = new_key;  /* sets new key */
  T->deg[T->root_new] = 0;        /* sets new degree to 0 */
  T->mark[T->root_new] = MP_HEAP_UNMARKED;

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
  //printf("T->key[T->root_new]: %d, T->child[T->root_new]: %d, T->m: %d\n", T->key[T->root_new], T->child[T->root_new], T->m);
}

__device__ void mp_heap_min_cuda_fibonacci_defragment
(
  MPHeapMin_Fibonacci *T
)
{
  MPInt r = 0;
  MPInt rc = 0;
  MPInt n_parsed = 0;
  MPInt curr = 0;
  MPInt i = 0;

  for (r = T->root_first; r != -1; r = T->next[r])
  {
    rc = r;
    while (rc > -1)
    {
      for (i = T->child[rc]; i != -1; i = T->next[i])
      {
        while ((curr != MP_HEAP_NULL) && (curr < n_parsed))
        {
          curr+=1;
        }

        if ((curr < n_parsed))
        {
          mp_heap_min_cuda_fibonacci_node_move(T, i, curr);
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

__device__ void mp_heap_min_cuda_fibonacci_node_move
(
  MPHeapMin_Fibonacci *T,
  MPInt source,
  MPInt dest
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
  T->mark[source] = MP_HEAP_NULL;

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

__device__ MPInt mp_heap_min_cuda_fibonacci_extract_min
(
  MPHeapMin_Fibonacci *T,
  MPInt *return_key
)
{
  MPInt return_index = -1;

  if (T->m > 0)
  {
    return_index = T->min_index;
    *return_key = T->key[T->min_index];
    mp_heap_min_cuda_fibonacci_delete_min(T);
  }
  else
  {
    *return_key = -1;
  }

  return return_index;
}

__device__ void mp_heap_min_cuda_fibonacci_delete_min
(
  MPHeapMin_Fibonacci *T
)
{
  MPInt i = 0;      /* parses nodes */
  MPInt next = -1;  /* for finding next root */

  #if DEBUG_HEAP == 1
    printf("\n     >>>>>>>>>>>> extracting [T->m: %d] <<<<<<<<<<<<\n", T->m);
    printf("\n ");
    //mp_heap_min_fibonacci_view_roots(T);
    printf("\n extracting node: %d\n", T->min_index);
    printf("T->min_index: %d -> T->key[T->min_index]: %d\n", T->min_index, T->key[T->min_index]);
    printf(" -> T->deg[%d]: %d\n", T->min_index, T->deg[T->min_index]);
    printf(" -> [root_first: %d, root_last: %d]\n\n", T->root_first, T->root_last);
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
        T->root_first = T->child[T->min_index]; /* sets child as first root */
      }

      {
        MPInt last_child = T->child[T->min_index];  /* moves to last child */
        MPInt c = 0;
        MPInt count = 0;
        #if DEBUG_HEAP == 1
          printf("T->min_index: %d\n", T->min_index);
          printf("  last_child: %d\n", last_child);
          printf("T->next[last_child]: %d\n", T->next[last_child]);
          printf("T->next[T->next[last_child]]: %d\n", T->next[T->next[last_child]]);
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
        MPInt last_child = T->child[T->min_index];  /* moves to last child */
        //while ((last_child = T->next[last_child]) != -1) {printf("last_child: %d\n", last_child);}
        while (T->next[last_child] != -1)
        {
          last_child = T->next[last_child];
        }

        T->previous[T->next[T->min_index]] = last_child;  /* previous of next[T->min] */
        T->next[last_child] = T->next[T->min_index];      /* next of last child */
      }
      else if (T->next[T->min_index] == -1)
      {
        MPInt last_child = T->child[T->min_index];  /* moves to last child */
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
      printf(" -> [root_first: %d, root_last: %d]\n\n", T->root_first, T->root_last);
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
      //mp_heap_min_fibonacci_view_roots(T);
    #endif

    //printf("intermediate\n");
    //mp_heap_min_fibonacci_view_roots(T);
    //printf("\n");
    //printf("T->min_index: %d -> %d\n", T->min_index, T->key[T->min_index]);
    //printf("T->next[T->min_index]: %d\n", T->next[T->min_index]);
    //printf("T->previous[T->min_index]: %d\n", T->previous[T->min_index]);
    //printf("\n");

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
        printf(" -> i: %d, root_first: %d, root_last: %d\n", i, T->root_first, T->root_last);
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
        MPInt parent = -1;
        MPInt child = -1;
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
          //mp_heap_min_fibonacci_view_roots(T);
          printf("root_first: %d, root_last: %d\n", T->root_first, T->root_last);
          printf(" ------?   T->parent[%d]: %d\n", i, T->parent[i]);
          printf(" ------?    T->child[%d]: %d\n", T->parent[i], T->child[T->parent[i]]);
          printf(" ------?     T->next[%d]: %d\n", T->previous[i], T->next[T->previous[i]]);
          printf(" ------? T->previous[%d]: %d\n", T->next[i], T->previous[T->next[i]]);
          printf("parent: %d -> %d\n", parent, T->key[parent]);
          printf(" child: %d -> %d\n", child, T->key[child]);
          printf("T->deg_mark[T->deg[i]]: %d -> %d\n", T->deg_mark[T->deg[i]], T->key[T->deg_mark[T->deg[i]]]);
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
          //mp_heap_min_fibonacci_view_roots(T);
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
          MPInt child_temp = T->child[parent];
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
          //mp_heap_min_fibonacci_view_roots(T);
          printf(">> T->deg_mark[T->deg[parent]+1]: %d\n", T->deg_mark[T->deg[parent]+1]);
          printf(">> T->deg[parent]+1: %d\n", T->deg[parent]+1);
          printf("parent: %d\n", parent);
          printf(" child: %d\n", child);
        #endif
        if ((T->deg_mark[T->deg[parent]+1] > -1) && (T->deg_mark[T->deg[parent]+1] != T->parent[child]))
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
        //mp_heap_min_fibonacci_view_roots(T);
        printf("last next: %d\n", next);
      #endif

      /* break when reaches root_last */
      if ((next != -1) || ((next != -1) && (next == T->parent[i])) || ((next != -1) && (T->deg[next] == T->deg[i])))
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
      //mp_heap_min_fibonacci_view_roots(T);
    #endif

    /*--------------------*/
    /* find new min_index */
    /*--------------------*/
    T->min_index = T->root_first;
    i = T->next[T->root_first];
    MPInt min_new = T->root_first;
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
    T->mark[T->min_index] = MP_HEAP_UNMARKED;
  }
  else if (T->m == 1)
  {
    T->mark[T->min_index] = MP_HEAP_UNMARKED;
    T->next[T->min_index] = -1;
    T->previous[T->min_index] = -1;
    T->min_index = -1;
    T->root_new = 0;
    T->root_first = -1;
    T->root_last = -1;
  }
  T->m -= 1;
}

__device__ void mp_heap_min_cuda_fibonacci_decrease
(
  MPHeapMin_Fibonacci *T,
  MPInt i,
  MPInt new_key
)
{
  MPInt j = 0;
  if (T->key[i] < new_key)
  {
    T->key[i] = new_key;
    T->mark[i] = MP_HEAP_UNMARKED;
    if (T->key[T->child[i]] < T->key[i])
    {
      j = T->parent[i];
      T->parent[i] = -1;
      T->next[T->root_last] = i;
      T->root_last = i;
      while (T->mark[j] == MP_HEAP_MARKED)
      {
        T->mark[j]= MP_HEAP_UNMARKED;
        T->next[T->root_last] = j;
        T->parent[j] = -1;
        T->previous[j] = T->root_last;
        T->root_last = j;
        j = T->parent[i];
      }
      if (T->parent[j] != -1)
      {
        T->mark[j] = MP_HEAP_MARKED;
      }
    }
  }
}

//__global__ void mp_cuda_blocking_psy_fA_kernel
//(
//  int m,
//  int blk_fA,
//  MPPatternCsr *P0, /*  input: sparse pattern */
//  MPPatternCsr *P1, /* output: coarse sparse pattern */
//  int *temp_array
//)
//{
//  int i = 0;
//  //int job_id = 0;
//
//  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
//
//  int blk = (1-(thread_id+1)/m)*blockIdx.x + ((thread_id+1)/m)*(m%blockIdx.x);
//  int start = thread_id;
//  //int end = start + blk;
//
//  int col = 0;
//  //int col_temp = 0;
//  int nz = 0;
//
//  int n_levels = m;
//  int term = 0;
//
//  /*                                                      */
//  /* PASS-1: computation of nz for a block of blk_fA rows */
//  /*                                                      */
//
//  for (i = 0; i < blk; ++i)
//  {
//    temp_array[i] = P0->cols[i];
//  }
//
//  do
//  {
//    term = 0;
//    for (i = 0; i < blk; ++i)
//    {
//      nz
//         = ((P0->cols[temp_array[i]]/blk_fA-col) > 0)*(nz+1)  /* new block */
//         + ((P0->cols[temp_array[i]]/blk_fA-col) == 0)*nz;    /* same block */
//
//      col = P0->cols[temp_array[i]]/blk_fA;
//      P1->rows_start[start+i] += 1;
//      term += (P0->rows_start[start+i+1]-temp_array[i]);
//    }
//  }
//  while(term);
//
//  //cudaDeviceSynchronize();
//
//  /* parallel partial sum algorithm to merge P1->rows_start array entries */
//
//  //n_levels = ceil((double) log2(m));  /* ERROR: calling C host function from CUDA function */
//  n_levels = 1; /* replace with the above */
//  for (i = 0; i < n_levels; ++i)
//  {
//    P1->rows_start[thread_id] = (thread_id%2)*P1->rows_start[thread_id-1] + ((thread_id+1)%2)*P1->rows_start[thread_id]; /* for perfectly divided m */
//  }
//
//  /*                                      */
//  /* PASS-2: evaluation of P1->cols array */
//  /*                                      */
//
//  //cudaDeviceSynchronize();
//
////  *nz_new = n_nodes_new;
//
//
////  MPPthreadContext_Probing *t_context = (MPPthreadContext_Probing*)t_context_packed;
////  int job_id = -1;
////  int next_job_id = -1;
////  int min_index;
////  int i = 0;
////  int count = 0;
////  int p = 0;
////
////  int m_B = t_context->m_B;
////  int blk = t_context->blk_fA;
////  int n_threads = t_context->n_threads;
////  int n_blocks = (m_B/blk+blk-1)/blk;
////  int m_P = m_B;
////
////  for (p = 0; p < t_context->n_levels; ++p)
////  {
////    /*----------------*/
////    /* initialization */
////    /*----------------*/
////    n_blocks = (m_P+blk-1)/blk;
////    m_P = m_P/blk;
////    t_context->m_P = t_context->m_P/blk;
////    for (i = 0; i < (n_blocks+n_threads-1)/n_threads; ++i)
////    {
////      mp_heap_min_fibonacci_insert(&t_context->heap, n_threads*i+t_context->mp_thread_id);
////    }
////
////    if (t_context->mp_thread_id)
////    {
////      t_context->shared->signals_continue[t_context->mp_thread_id] = 0;
////    }
////
////    /*------------*/
////    /* coarsening */
////    /*------------*/
//    //mp_heap_min_fibonacci_extract_min(&t_context->heap, &job_id);
//    //while (job_id > -1)
//    //{
////      /* applies coarsening */
////      mp_pthread_blocking_contract
////      (
////        t_context->m_P,
////        t_context->blk_fA,
////        &t_context->nz,
////        t_context->blk_fA*job_id,
////        t_context->blk_fA*(job_id+1),
////        t_context->P0,
////        t_context->P1,
////        t_context->temp_array,
////        t_context->temp_inverted_array,
////        t_context->temp_cols
////      );
////
////      /*  merging computed row in temp_cols to P1->cols (ordering required) */
////      pthread_mutex_lock(&mutex_merge);
////      while (t_context->shared->max_priority_id != t_context->mp_thread_id)
////      {
////        pthread_cond_wait(&cond, &mutex_merge);
////      }
////
////      /* updates P1->rows_start and P1->rows_end */
////      t_context->P1->rows_start[job_id] = t_context->shared->nz;
////      t_context->P1->rows_end[job_id]
////        = t_context->P1->rows_start[job_id] + t_context->nz;
////
////      //if (job_id < 3)
////      //{
////      //  printf("tid: %d, job_id: %d, p1_re: %d\n", t_context->mp_thread_id, job_id, t_context->P1->rows_end[job_id]);
////      //}
////
////      /* reallocates memory if it is required */
////      if (t_context->shared->nz + t_context->nz >= t_context->P1->nz_max)
////      {
////        t_context->P1->nz_max += t_context->mem_inc;
////        t_context->P1->cols = mkl_realloc(t_context->P1->cols, sizeof(int)*t_context->P1->nz_max);
////      }
////
////      /* copies columns from temp_cols to P1->cols */
////      for (i = 0; i < t_context->nz; ++i)
////      {
////        t_context->P1->cols[t_context->shared->nz+i] = t_context->temp_cols[i];
////      }
////      count += 1; /* for debugging only */
////      t_context->shared->nz += t_context->nz; /* global nonzeros */
////
////      /* static scheduling */
////      mp_heap_min_fibonacci_extract_min(&t_context->heap, &job_id);
////      t_context->shared->max_priority_id
////        = (t_context->shared->max_priority_id + 1) % t_context->n_threads;
////      pthread_cond_broadcast(&cond);
////      pthread_mutex_unlock(&mutex_merge);
////    }
////
////    //printf("t_context->P1->rows_start[0]: %d -  %d\n", t_context->P1->rows_start[0], t_context->P1->rows_end[0]);
////    //printf("t_context->P1->rows_start[1]: %d -  %d\n", t_context->P1->rows_start[1], t_context->P1->rows_end[1]);
////    //printf("t_context->P1->rows_start[2]: %d -  %d\n", t_context->P1->rows_start[2], t_context->P1->rows_end[2]);
////
////    pthread_barrier_wait(&barrier); //may not be required
////
////    pthread_mutex_lock(&mutex);
////    while ((t_context->mp_thread_id) && (!t_context->shared->signals_continue[t_context->mp_thread_id]))
////    {
////      pthread_cond_wait(&cond, &mutex);
////    }
////
////    if (t_context->mp_thread_id == 0)
////    {
////      /* synchronizes shared structures */
////      t_context->P1->m = t_context->P1->m/t_context->blk_fA;
////      t_context->P1->nz = t_context->shared->nz;
////      t_context->P0->m = t_context->m_P;
////      t_context->shared->nz = 0;
////      t_context->shared->max_priority_id = 0;
////      for (i = 1; i < t_context->n_threads; ++i)
////      {
////        t_context->shared->signals_continue[i] = 1;
////      }
////
////      pthread_cond_broadcast(&cond);
////    }
////    else
////    {
////      t_context->shared->signals_continue[t_context->mp_thread_id] = 0;
////    }
////    pthread_mutex_unlock(&mutex);
////
////    /* updates for all threads */
////    {
////      //((n_blocks+n_threads-1)/n_threads)*t_context->mp_thread_id+i
////      for (i = 0; i < (n_blocks+n_threads-1)/n_threads; ++i)
////      {
////        mp_heap_min_fibonacci_insert(&t_context->heap, n_threads*i+t_context->mp_thread_id);
////      }
////    }
////    memcpy(t_context->temp_array, t_context->P1->rows_start, (sizeof *t_context->temp_array)*t_context->m_P);
////
////    //printf("--------- before expand ---------\n");
////    //printf("t_context->P0->m_P: %d\n", t_context->P0->m);
////    //printf("t_context->P1->m_P: %d\n", t_context->P1->m);
////    //printf("t_context->P1->rows_start[0]: %d -  %d\n", t_context->P0->rows_start[0],t_context->P1->rows_end[0] );
////    //printf("t_context->P1->rows_start[1]: %d -  %d\n", t_context->P0->rows_start[1],t_context->P1->rows_end[1] );
////
////    /*--------------------*/
////    /* sparse mm (expand) */
////    /*--------------------*/
////    count = 0;
////    mp_heap_min_fibonacci_extract_min(&t_context->heap, &job_id);
////    while (job_id > -1)
////    {
////      //printf("(first) tid: %d, job_id: %d\n", t_context->mp_thread_id, job_id);
////      //printf("t_context->m_P: %d\n", t_context->m_P);
////      //printf("tid: %d, job_id: %d -> t_context->blk_fA*job_id: %d\n", t_context->mp_thread_id, job_id, t_context->blk_fA*job_id);
////      //printf("t_context->P1->m: %d\n",  t_context->P1->m);
////      mp_pthread_psy_expand_distance_22
////      (
////        t_context->m_P,
////        t_context->blk_fA,
////        &t_context->nz,
////        t_context->blk_fA*job_id,
////        t_context->blk_fA*(job_id+1),
////        t_context->P1,
////        t_context->P0,
////        t_context->temp_array,
////        t_context->temp_inverted_array,
////        t_context->temp_cols
////      );
////
////      //printf("(inter)\n");
////
////      /*  merging computed row in temp_cols to P1->cols */
////      pthread_mutex_lock(&mutex_merge);
////      while (t_context->shared->max_priority_id != t_context->mp_thread_id)
////      {
////        pthread_cond_wait(&cond, &mutex_merge);
////      }
////
////      /* computes P0->rows_end */
////      for (i = 0; i < t_context->blk_fA; ++i)
////      {
////        t_context->P0->rows_end[t_context->blk_fA*job_id+i] = t_context->shared->nz + t_context->temp_inverted_array[t_context->blk_fA*job_id+i];
////      }
////
////      /* computes P0->rows_start */
////      t_context->P0->rows_start[t_context->blk_fA*job_id] = t_context->shared->nz;
////      for (i = 1; i < t_context->blk_fA; ++i)
////      {
////        t_context->P0->rows_start[t_context->blk_fA*job_id+i] = t_context->P0->rows_end[t_context->blk_fA*job_id+i-1];
////      }
////
////      /* reallocates if it is required */
////      if (t_context->shared->nz + t_context->nz >= t_context->P0->nz_max)
////      {
////        t_context->P0->nz_max += t_context->mem_inc;
////        t_context->P0->cols = mkl_realloc(t_context->P0->cols, sizeof(int)*t_context->P0->nz_max);
////      }
////
////      /* copies columns from P0->cols to temp_cols */
////      for (i = 0; i < t_context->nz; ++i)
////      {
////        t_context->P0->cols[t_context->shared->nz+i] = t_context->temp_cols[i];
////      }
////
////      t_context->shared->nz += t_context->nz; /* updates shared->nz */
////      count += 1;
////
////      /* static scheduling */
////      mp_heap_min_fibonacci_extract_min(&t_context->heap, &job_id);
////      t_context->shared->max_priority_id
////        = (t_context->shared->max_priority_id + 1) % t_context->n_threads;
////
////      pthread_cond_broadcast(&cond);
////      pthread_mutex_unlock(&mutex_merge);
////      //printf("(last) tid: %d, job_id: %d\n", t_context->mp_thread_id, job_id);
////    }
////
////    //printf("tid: %d, after expand\n", t_context->mp_thread_id);
////    //printf("t_context->P0->rows_start[0]: %d - %d\n", t_context->P0->rows_start[0],t_context->P0->rows_end[0]);
////    //printf("t_context->P0->rows_start[1]: %d - %d\n", t_context->P0->rows_start[1],t_context->P0->rows_end[1]);
////    //printf("t_context->P0->rows_start[2]: %d - %d\n", t_context->P0->rows_start[2],t_context->P0->rows_end[2]);
////    //printf("t_context->P0->rows_start[3]: %d - %d\n", t_context->P0->rows_start[3],t_context->P0->rows_end[3]);
////    //printf("t_context->P0->rows_start[4]: %d - %d\n", t_context->P0->rows_start[4],t_context->P0->rows_end[4]);
////    //printf("P1->m: %d\n", t_context->P1->m);
////
////    pthread_barrier_wait(&barrier);
////
////    pthread_mutex_lock(&mutex);
////    while ((t_context->mp_thread_id) && (!t_context->shared->signals_continue[t_context->mp_thread_id]))
////    {
////      pthread_cond_wait(&cond, &mutex);
////    }
////
////    if (t_context->mp_thread_id == 0)
////    {
////      /* synchronizes shared structures */
////      t_context->P0->nz = t_context->shared->nz;
////      t_context->shared->nz = 0;
////      t_context->shared->max_priority_id = 0;
////      //t_context->shared->continue_signal = 1;
////      for (i = 1; i < t_context->n_threads; ++i)
////      {
////        t_context->shared->signals_continue[i] = 1;
////      }
////      pthread_cond_broadcast(&cond);
////    }
////    else
////    {
////      t_context->shared->signals_continue[t_context->mp_thread_id] = 0;
////    }
////
////    pthread_mutex_unlock(&mutex);
////  }
//
//}

//__global__ void prefix_scan
//(
//  int m,    /* number of entries */
//  float *v, /* input vector */
//  float *w  /* output vector */
//)
//{
//  extern __shared__ float mem_shared[];
//
//  int thread_id = threadIdx.x;
//
//  int stride_offset = 1; /* increments in different levels */
//
//  int l = 0; /* left */
//  int r = 0; /* right */
//
//  float t = 0.0; /* temp */
//
//  mem_shared[2*thread_id] = v[2*thread_id];
//  mem_shared[2*thread_id+1] = v[2*thread_id+1];
//
//
//  /*-------------------*/
//  /* PHASE 1: up sweep */
//  /*-------------------*/
//  int m_curr = m;
//  while (m_curr > 0)
//  {
//    __syncthreads();  /* wait for all threads in current block */
//    if (thread_id < m_curr)
//    {
//      l = (stride_offset*2)*thread_id + (stride_offset-1);
//      r = (stride_offset*2)*thread_id + (2*stride_offset-1);
//      mem_shared[r] += mem_shared[l];
//      //printf("i: %d/%d, thread_id: %d, [l: %d, r: %d]\n", i, m_curr, thread_id, l, r);
//    }
//
//    stride_offset = stride_offset*2;  /* offset increases exponentially */
//    m_curr = m_curr/2;  /* range decreases exponentially */
//  }
//
//  __syncthreads();  // (!) added by me
//
//
//  /*---------------------*/
//  /* PHASE 2: down sweep */
//  /*---------------------*/
//
//  if (thread_id == 0)
//  {
//    mem_shared[m] = mem_shared[m-1];
//    mem_shared[m-1] = 0; /* clears last element */
//  }
//
//  m_curr = 1;
//  stride_offset = m;
//  while (m_curr < m)
//  {
//    stride_offset = stride_offset/2;
//    __syncthreads();
//    if (thread_id < m_curr)
//    {
//      l = (stride_offset*2)*thread_id + (stride_offset-1);
//      r = (stride_offset*2)*thread_id + (2*stride_offset-1);
//      t = mem_shared[l];
//      mem_shared[l] = mem_shared[r];
//      mem_shared[r] += t;
//      //printf(" - i: %d/%d, thread_id: %d, [l: %d, r: %d]\n", i, m_curr, thread_id, l, r);
//    }
//    m_curr = m_curr*2;
//  }
//  __syncthreads();
//
//  /* writes output to device memory */
//  w[2*thread_id] = mem_shared[2*thread_id+1];
//  w[2*thread_id+1] = mem_shared[2*thread_id+2];
//}

__global__ void mp_i_prefix_scan
(
  MPInt m,  /* number of entries */
  MPInt *v, /* input vector */
  MPInt *w  /* output vector */
)
{
  extern __shared__ MPInt mem_shared_i[];

  MPInt thread_id = threadIdx.x;

  MPInt stride_offset = 1; /* increments in different levels */

  MPInt l = 0; /* left */
  MPInt r = 0; /* right */
  MPInt t = 0; /* temp */

  mem_shared_i[2*thread_id] = v[2*thread_id];
  mem_shared_i[2*thread_id+1] = v[2*thread_id+1];

  /*-------------------*/
  /* PHASE 1: up sweep */
  /*-------------------*/
  if (thread_id < m)
  {
    MPInt m_curr = m;
    while (m_curr > 0)
    {
      __syncthreads();  /* wait for all threads in current block */
      if (thread_id < m_curr)
      {
        l = (stride_offset*2)*thread_id + (stride_offset-1);
        r = (stride_offset*2)*thread_id + (2*stride_offset-1);
        mem_shared_i[r] += mem_shared_i[l];
        //printf("i: %d/%d, thread_id: %d, [l: %d, r: %d]\n", i, m_curr, thread_id, l, r);
      }

      stride_offset = stride_offset*2;  /* offset increases exponentially */
      m_curr = m_curr/2;  /* range decreases exponentially */
    }

    __syncthreads();  // (!) added by me


    /*---------------------*/
    /* PHASE 2: down sweep */
    /*---------------------*/

    if (thread_id == 0)
    {
      mem_shared_i[m] = mem_shared_i[m-1];
      mem_shared_i[m-1] = 0; /* clears last element */
    }

    m_curr = 1;
    stride_offset = m;
    while (m_curr < m)
    {
      stride_offset = stride_offset/2;
      __syncthreads();
      if (thread_id < m_curr)
      {
        l = (stride_offset*2)*thread_id + (stride_offset-1);
        r = (stride_offset*2)*thread_id + (2*stride_offset-1);
        t = mem_shared_i[l];
        mem_shared_i[l] = mem_shared_i[r];
        mem_shared_i[r] += t;
      }
      m_curr = m_curr*2;
    }
    __syncthreads();

    /* writes output to device memory */
    w[2*thread_id] = mem_shared_i[2*thread_id+1];
    w[2*thread_id+1] = mem_shared_i[2*thread_id+2];
  }
}

//__device__ void prefix_scan_device
//(
//  int thread_id,
//  int m,   /* number of entries */
//  int *v,  /* input vector */
//  int *mem_shared
//)
//{
//  int stride_offset = 1; /* increments in different levels */
//
//  int l = 0; /* left */
//  int r = 0; /* right */
//
//  float t = 0.0; /* temp */
//
//  mem_shared[2*thread_id] = v[2*thread_id];
//  mem_shared[2*thread_id+1] = v[2*thread_id+1];
//
//
//  /*-------------------*/
//  /* PHASE 1: up sweep */
//  /*-------------------*/
//  int m_curr = m;
//  while (m_curr > 0)
//  {
//    __syncthreads();  /* wait for all threads in current block */
//    if (thread_id < m_curr)
//    {
//      l = (stride_offset*2)*thread_id + (stride_offset-1);
//      r = (stride_offset*2)*thread_id + (2*stride_offset-1);
//      mem_shared[r] += mem_shared[l];
//      //printf("i: %d/%d, thread_id: %d, [l: %d, r: %d]\n", i, m_curr, thread_id, l, r);
//    }
//
//    stride_offset = stride_offset*2;  /* offset increases exponentially */
//    m_curr = m_curr/2;  /* range decreases exponentially */
//  }
//
//  __syncthreads();  // (!) added by me
//
//
//  /*---------------------*/
//  /* PHASE 2: down sweep */
//  /*---------------------*/
//
//  if (thread_id == 0)
//  {
//    mem_shared[m] = mem_shared[m-1];
//    mem_shared[m-1] = 0; /* clears last element */
//  }
//
//  m_curr = 1;
//  stride_offset = m;
//  while (m_curr < m)
//  {
//    stride_offset = stride_offset/2;
//    __syncthreads();
//    if (thread_id < m_curr)
//    {
//      l = (stride_offset*2)*thread_id + (stride_offset-1);
//      r = (stride_offset*2)*thread_id + (2*stride_offset-1);
//      t = mem_shared[l];
//      mem_shared[l] = mem_shared[r];
//      mem_shared[r] += t;
//      //printf(" - i: %d/%d, thread_id: %d, [l: %d, r: %d]\n", i, m_curr, thread_id, l, r);
//    }
//    m_curr = m_curr*2;
//  }
//}

__global__ void mp_i_prefix_scan_uneven
(
  int m_gl,  /* number of entries */
  int *v, /* input vector */
  int *w,  /* output vector */
  int *acc
)
{
  extern __shared__ int mem_shared[];

  int thread_id = threadIdx.x;
  int thread_id_gl = blockDim.x*blockIdx.x + threadIdx.x;;


  int s = 1; /* increments in different levels */

  int l = 0; /* left */
  int r = 0; /* right */

  int t = 0; /* temp */
  int count = 0;

  int nz = 0;
  //int m = m_gl - blockDim.x*blockIdx.x*2;
  int m = (1-(blockIdx.x+gridDim.x-1)/gridDim.x)*(2*blockDim.x) + ((blockIdx.x+gridDim.x-1)/gridDim.x)*(m_gl-(gridDim.x-1)*(2*blockDim.x));

  mem_shared[2*thread_id] = v[2*thread_id_gl];
  mem_shared[2*thread_id+1] = v[2*thread_id_gl+1];

  /*-------------------*/
  /* PHASE 1: up sweep */
  /*-------------------*/
  if (thread_id < m)
  {
    //printf("gridDim.x: %d, blockIdx.x: %d, m: %d, (m_gl%(2*blockDim.x)): %d\n", gridDim.x, blockIdx.x, m, (m_gl%(2*blockDim.x)));
    int m_curr = m;
    while (m_curr > 0)
    {
      __syncthreads();
      if (thread_id < m_curr)
      {
        l = s*2*thread_id + (s-1);
        r = s*2*thread_id + (2*s-1);

        if ((l >= m-1) && (r > m-1))
        {}
        else if ((l < m-1) && (r > m-1))
        {
          r = m-1;
          mem_shared[r] += mem_shared[l];
          //printf("*|-> m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d\n", m_curr, s, thread_id, l, r, mem_shared[r]);
        }
        else if ((l < m-1) && (r <= m-1))
        {
          r = s*2*thread_id + (2*s-1);
          mem_shared[r] += mem_shared[l];
          //printf("|-> m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d\n", m_curr, s, thread_id, l, r, mem_shared[r]);
        }
      }
      s = s*2;
      m_curr = m_curr/2;
      count += 1;
    }
    __syncthreads();


    //if (blockIdx.x == 7)
    //{
    //  printf("(after) blockIdx.x: %d, YOOOOOOOOOOOOOOOOO\n", blockIdx.x);
    //}

    /*---------------------*/
    /* PHASE 2: down sweep */
    /*---------------------*/

    //if ((thread_id == 0) && (m > 3))
    if (thread_id == 0)
    {
      acc[blockIdx.x] = mem_shared[m-1];
      nz = mem_shared[m-1];
      //printf(">> blkIdx.x: %d, acc[0]: %d, acc[1]: %d, mem_shared[m-1]: %d\n", blockIdx.x, acc[0], acc[1], mem_shared[m-1]);
      //acc[blockIdx.x] = 2048;
      mem_shared[m] = mem_shared[m-1];
      mem_shared[m-1] = 0; /* clears last element */
      //printf("  m: %d, blockIdx.x: %d <-| m-1: %d, mem_shared[m-1]: %d, s: %d, acc[blockIdx.x]: %d\n", m, blockIdx.x, m-1, mem_shared[m-1], s, acc[blockIdx.x]);
      if (blockIdx.x == 7)
      {
        printf("acc[blockIdx.x]: %d\n", acc[blockIdx.x]);
      }
    }

    //if (acc[0] < 4000)
    //{
    //  printf("blockIdx.x: %d, thread_id: %d, acc[0]: %d\n", blockIdx.x, thread_id, acc[0]);
    //}

    int m_max = s;
    m_curr = 1;
    s = s/2;
    int jc = 0;

    while ((count > 0) && (m > 3))
    {
      __syncthreads();

      if ((thread_id < m_curr) && (m_curr == 1))
      {
        l = (m_max)/2-1;
        r = m-1;
        t = mem_shared[l];
        if (l < r)
        {
          mem_shared[l] = mem_shared[r];
          mem_shared[r] += t;
          //printf("<-| m_max: %d -> l: %d\n", m_max, l);
          //printf("<-| m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d, t: %d\n", m_curr, s, thread_id, l, r, mem_shared[r], t);
        }
      }
      else if (thread_id < m_curr)
      {
        //printf("here: %d (thread_id)\n", thread_id);
        l = (s*2)*thread_id + (s-1);
        r = (s*2)*thread_id + (2*s-1);
        if (r > m-1)
        {
          r = m-1;
        }
        t = mem_shared[l];
        if (l < r)
        {
          mem_shared[l] = mem_shared[r];
          mem_shared[r] += t;
          //printf("  m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d]\n", m_curr, s, thread_id, l, r);
        }
      }
      m_curr = m_curr*2;
      s = s/2;
      count -= 1;
      //if ((acc[0] > 2999) && (jc == 0))
      //{
      //  //printf("acc[0]: %d, thread_id_gl: %d\n", acc[0], thread_id_gl);
      //  jc += 1;
      //}
    }
    __syncthreads();

    //if (acc[0] < 4000)
    //{
    //  printf("blockIdx.x: %d, thread_id: %d, acc[0]: %d\n", blockIdx.x, thread_id, acc[0]);
    //}

    /* writes output to device memory (transposed by 1 output) */
      //printf(">> acc[0]: %d, acc[1]: %d\n", acc[0], acc[1]);
      w[2*thread_id_gl] = mem_shared[2*thread_id+1];
      w[2*thread_id_gl+1] = mem_shared[2*thread_id+2];
  }
}

__global__ void mp_i_prefix_scan_uneven_off
(
  int m_gl,  /* number of entries */
  int *v, /* input vector */
  int *w,  /* output vector */
  int *acc
)
{
  extern __shared__ int mem_shared_off[];

  int thread_id = threadIdx.x;
  int thread_id_gl = blockDim.x*blockIdx.x + threadIdx.x;;


  int s = 1; /* increments in different levels */

  int l = 0; /* left */
  int r = 0; /* right */

  int t = 0; /* temp */
  int count = 0;

  int nz = 0;
  //int m = m_gl - blockDim.x*blockIdx.x*2;
  int m = (1-(blockIdx.x+gridDim.x-1)/gridDim.x)*(2*blockDim.x) + ((blockIdx.x+gridDim.x-1)/gridDim.x)*(m_gl-(gridDim.x-1)*(2*blockDim.x));

  mem_shared_off[2*thread_id] = v[2*thread_id_gl];
  mem_shared_off[2*thread_id+1] = v[2*thread_id_gl+1];

  /*-------------------*/
  /* PHASE 1: up sweep */
  /*-------------------*/
  if (thread_id < m)
  {
    //printf("gridDim.x: %d, blockIdx.x: %d, m: %d, (m_gl%(2*blockDim.x)): %d\n", gridDim.x, blockIdx.x, m, (m_gl%(2*blockDim.x)));
    int m_curr = m;
    while (m_curr > 0)
    {
      __syncthreads();
      if (thread_id < m_curr)
      {
        l = s*2*thread_id + (s-1);
        r = s*2*thread_id + (2*s-1);

        if ((l >= m-1) && (r > m-1))
        {}
        else if ((l < m-1) && (r > m-1))
        {
          r = m-1;
          mem_shared_off[r] += mem_shared_off[l];
          //printf("*|-> m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d\n", m_curr, s, thread_id, l, r, mem_shared[r]);
        }
        else if ((l < m-1) && (r <= m-1))
        {
          r = s*2*thread_id + (2*s-1);
          mem_shared_off[r] += mem_shared_off[l];
          //printf("|-> m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d\n", m_curr, s, thread_id, l, r, mem_shared[r]);
        }
      }
      s = s*2;
      m_curr = m_curr/2;
      count += 1;
    }
    __syncthreads();


    //if (blockIdx.x == 7)
    //{
    //  printf("(after) blockIdx.x: %d, YOOOOOOOOOOOOOOOOO\n", blockIdx.x);
    //}

    /*---------------------*/
    /* PHASE 2: down sweep */
    /*---------------------*/

    //if ((thread_id == 0) && (m > 3))
    if (thread_id == 0)
    {
      acc[blockIdx.x] = mem_shared_off[m-1];
      nz = mem_shared_off[m-1];
      //printf(">> blkIdx.x: %d, acc[0]: %d, acc[1]: %d, mem_shared[m-1]: %d\n", blockIdx.x, acc[0], acc[1], mem_shared[m-1]);
      //acc[blockIdx.x] = 2048;
      mem_shared_off[m] = mem_shared_off[m-1];
      mem_shared_off[m-1] = 0; /* clears last element */
      //printf("  m: %d, blockIdx.x: %d <-| m-1: %d, mem_shared[m-1]: %d, s: %d, acc[blockIdx.x]: %d\n", m, blockIdx.x, m-1, mem_shared[m-1], s, acc[blockIdx.x]);
      if (blockIdx.x == 7)
      {
        printf("acc[blockIdx.x]: %d\n", acc[blockIdx.x]);
      }
    }

    //if (acc[0] < 4000)
    //{
    //  printf("blockIdx.x: %d, thread_id: %d, acc[0]: %d\n", blockIdx.x, thread_id, acc[0]);
    //}

    int m_max = s;
    m_curr = 1;
    s = s/2;
    int jc = 0;

    while ((count > 0) && (m > 3))
    {
      __syncthreads();

      if ((thread_id < m_curr) && (m_curr == 1))
      {
        l = (m_max)/2-1;
        r = m-1;
        t = mem_shared_off[l];
        if (l < r)
        {
          mem_shared_off[l] = mem_shared_off[r];
          mem_shared_off[r] += t;
          //printf("<-| m_max: %d -> l: %d\n", m_max, l);
          //printf("<-| m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d, t: %d\n", m_curr, s, thread_id, l, r, mem_shared[r], t);
        }
      }
      else if (thread_id < m_curr)
      {
        //printf("here: %d (thread_id)\n", thread_id);
        l = (s*2)*thread_id + (s-1);
        r = (s*2)*thread_id + (2*s-1);
        if (r > m-1)
        {
          r = m-1;
        }
        t = mem_shared_off[l];
        if (l < r)
        {
          mem_shared_off[l] = mem_shared_off[r];
          mem_shared_off[r] += t;
          //printf("  m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d]\n", m_curr, s, thread_id, l, r);
        }
      }
      m_curr = m_curr*2;
      s = s/2;
      count -= 1;
      //if ((acc[0] > 2999) && (jc == 0))
      //{
      //  //printf("acc[0]: %d, thread_id_gl: %d\n", acc[0], thread_id_gl);
      //  jc += 1;
      //}
    }
    __syncthreads();

    //if (acc[0] < 4000)
    //{
    //  printf("blockIdx.x: %d, thread_id: %d, acc[0]: %d\n", blockIdx.x, thread_id, acc[0]);
    //}

    /* writes output to device memory (transposed by 1 output) */
      //printf(">> acc[0]: %d, acc[1]: %d\n", acc[0], acc[1]);
      w[2*thread_id_gl] = mem_shared_off[2*thread_id];
      w[2*thread_id_gl+1] = mem_shared_off[2*thread_id+1];
  }
}

__global__ void mp_partial_sums_join
(
  int n_V,
  int n_acc,
  int *v,
  int *acc
)
{
  int i = 0;
  int j = 0;
  int thread_id_gl = blockDim.x*blockIdx.x + threadIdx.x;

  if (thread_id_gl < n_V)
  {
    for (i = 0; i < blockIdx.x/2; ++i)
    {
      //printf("acc[0]: %d, acc[1]: %d, acc[2]: %d, acc[3]: %d, acc[4]: %d, acc[5]: %d, acc[6]: %d, acc[7]: %d\n", acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
      //printf("i: %d, acc[0]: %d, acc[1]: %d\n", i, acc[0], acc[1]);
      v[thread_id_gl] += acc[i];
    }
  }
}

__host__ void mp_cuda_d_parsum
(
  int n_threads,  /* n_threads per block */
  int n,
  int *d_v_in,
  int *d_v_out,
  int *d_acc
)
{
  dim3 cuda_threads_per_block(n_threads, 1, 1);
  dim3 cuda_blocks_per_grid(((n+1)/2+cuda_threads_per_block.x-1)/cuda_threads_per_block.x, 1, 1);

  /* remains to do synchronization on top of blocks */
  mp_i_prefix_scan_uneven<<<cuda_blocks_per_grid, cuda_threads_per_block, sizeof(int)*(cuda_threads_per_block.x*2+1)>>>
  (
    n,      /* number of inputs left and right */
    d_v_in,    /* input vector */
    d_v_out,   /* output vector */
    d_acc
  );
  cudaDeviceSynchronize();

  cuda_blocks_per_grid = (n+cuda_threads_per_block.x-1)/cuda_threads_per_block.x;
  mp_partial_sums_join<<<cuda_blocks_per_grid, cuda_threads_per_block>>>
  (
    n,
    cuda_blocks_per_grid.x,
    d_v_out,
    d_acc
  );
}

__host__ void mp_cuda_d_parsum_off
(
  int n_threads,  /* n_threads per block */
  int n,
  int *d_v_in,
  int *d_v_out,
  int *d_acc
)
{
  dim3 cuda_threads_per_block(n_threads, 1, 1);
  dim3 cuda_blocks_per_grid(((n+1)/2+cuda_threads_per_block.x-1)/cuda_threads_per_block.x, 1, 1);

  /* remains to do synchronization on top of blocks */
  mp_i_prefix_scan_uneven_off<<<cuda_blocks_per_grid, cuda_threads_per_block, sizeof(int)*(cuda_threads_per_block.x*2+1)>>>
  (
    n,      /* number of inputs left and right */
    d_v_in,    /* input vector */
    d_v_out,   /* output vector */
    d_acc
  );
  cudaDeviceSynchronize();

  cuda_blocks_per_grid = (n+cuda_threads_per_block.x-1)/cuda_threads_per_block.x;
  mp_partial_sums_join<<<cuda_blocks_per_grid, cuda_threads_per_block>>>
  (
    n,
    cuda_blocks_per_grid.x,
    &d_v_out[1],
    d_acc
  );
}

//__global__ void mp_i_prefix_scan_uneven
//(
//  int m,  /* number of entries */
//  int *v, /* input vector */
//  int *w  /* output vector */
//)
//{
//  extern __shared__ int mem_shared[];
//
//  int thread_id = threadIdx.x;
//
//  int s = 1; /* increments in different levels */
//
//  int l = 0; /* left */
//  int r = 0; /* right */
//
//  int t = 0; /* temp */
//  int count = 0;
//
//  mem_shared[2*thread_id] = v[2*thread_id];
//  mem_shared[2*thread_id+1] = v[2*thread_id+1];
//
//  /*-------------------*/
//  /* PHASE 1: up sweep */
//  /*-------------------*/
//  if (thread_id < m)
//  {
//    int m_curr = m;
//    while (m_curr > 0)
//    {
//      __syncthreads();
//      if (thread_id < m_curr)
//      {
//        l = s*2*thread_id + (s-1);
//        r = s*2*thread_id + (2*s-1);
//
//        if ((l >= m-1) && (r > m-1))
//        {
//        }
//        else if (r > m-1)
//        {
//          r = m-1;
//          mem_shared[r] += mem_shared[l];
//          //printf("*|-> m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d\n", m_curr, s, thread_id, l, r, mem_shared[r]);
//        }
//        else
//        {
//          r = s*2*thread_id + (2*s-1);
//          mem_shared[r] += mem_shared[l];
//          //printf("|-> m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d\n", m_curr, s, thread_id, l, r, mem_shared[r]);
//        }
//      }
//      s = s*2;
//      m_curr = m_curr/2;
//      count += 1;
//    }
//    __syncthreads();
//
//
//    /*---------------------*/
//    /* PHASE 2: down sweep */
//    /*---------------------*/
//
//    if ((thread_id == 0) && (m > 3))
//    {
//      mem_shared[m] = mem_shared[m-1];
//      mem_shared[m-1] = 0; /* clears last element */
//      printf("<-| m-1: %d, mem_shared[m-1]: %d, s: %d\n", m-1, mem_shared[m-1], s);
//    }
//
//    int m_max = s;
//    m_curr = 1;
//    s = s/2;
//
//    while ((count > 0) && (m > 3))
//    {
//      __syncthreads();
//
//      if ((thread_id < m_curr) && (m_curr == 1))
//      {
//        l = (m_max)/2-1;
//        r = m-1;
//        t = mem_shared[l];
//        mem_shared[l] = mem_shared[r];
//        mem_shared[r] += t;
//        printf("<-| m_max: %d -> l: %d\n", m_max, l);
//        printf("<-| m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d] -> %d, t: %d\n", m_curr, s, thread_id, l, r, mem_shared[r], t);
//      }
//      else if (thread_id < m_curr)
//      {
//        l = (s*2)*thread_id + (s-1);
//        r = (s*2)*thread_id + (2*s-1);
//        if (r > m-1)
//        {
//          r = m-1;
//        }
//        t = mem_shared[l];
//        if (l < r)
//        {
//          mem_shared[l] = mem_shared[r];
//          mem_shared[r] += t;
//        }
//        //printf("  m_curr: %d, s: %d, thread_id: %d, [l: %d, r: %d]\n", m_curr, s, thread_id, l, r);
//      }
//      m_curr = m_curr*2;
//      s = s/2;
//      count -= 1;
//    }
//    __syncthreads();
//
//    /* writes output to device memory (transposed by 1 output) */
//    w[2*thread_id] = mem_shared[2*thread_id+1];
//    w[2*thread_id+1] = mem_shared[2*thread_id+2];
//
//    printf("thread_id: %d, w[2*thread_id]: %d, v[2*thread_id]: %d\n", thread_id, w[2*thread_id], v[2*thread_id]);
//
//    /* actual output (used for debug) */
//    //w[2*thread_id] = mem_shared[2*thread_id];
//    //w[2*thread_id+1] = mem_shared[2*thread_id+1];
//  }
//}

__host__ void mp_cuda_blocking_psy_fA_2
(
  MPContext *context
)
{
  MPInt p = 0;
  MPInt n_threads = context->n_threads_cuda_probing;
  MPInt n_levels = context->n_levels;
  MPInt blk = context->blk_fA;
  MPPatternCsr *h_P0 = &context->pattern_array.blocking[0];
  MPPatternCsr *h_P1 = &context->pattern_array.blocking[1];
  MPInt *temp_array = NULL; /* temporary device memory */

  MPInt m = h_P0->m;  /* dimension of input pattern */

  MPPatternCsr_Cuda P0;
  MPPatternCsr_Cuda P1;

  P0.m = h_P0->m;
  P0.nz = h_P0->nz;

  P1.m = h_P1->m;
  P1.nz = h_P1->nz;

  /* allocates device memory for P0 and P1 */
  cudaMalloc((void**)&P0.d_row_pointers, sizeof(MPInt)*(h_P0->m+1));
  cudaMalloc((void**)&P0.d_cols, sizeof(MPInt)*h_P0->nz*2);  /* overallocate */
  cudaMalloc((void**)&P1.d_row_pointers, sizeof(MPInt)*(h_P0->m+1));
  cudaMalloc((void**)&P1.d_cols, sizeof(MPInt)*h_P0->nz*2);  /* overallocate */
  cudaMalloc((void**)&temp_array, sizeof(MPInt)*(h_P0->nz+1));

  MPInt *d_acc = NULL;
  MPInt n_acc = ((m+1)/2+n_threads-1)/n_threads;
  cudaMalloc((void**)&d_acc, sizeof(MPInt)*(n_acc));

  printf("~~ after allocation: %d -> %s\n", cudaGetLastError(),
    cudaGetErrorName(cudaGetLastError()));

  /* initialization of P0 */
  cudaMemset(P0.d_row_pointers, 0, sizeof(MPInt));

  printf("~~ after memset: %d -> %s\n", cudaGetLastError(),
    cudaGetErrorName(cudaGetLastError()));
  cudaMemcpy(&P0.d_row_pointers[1], h_P0->rows_end, sizeof(MPInt)*m, cudaMemcpyHostToDevice);

  printf("~~ after memcpy P0.rows: %d -> %s\n", cudaGetLastError(),
    cudaGetErrorName(cudaGetLastError()));

  cudaMemcpy(P0.d_cols, h_P0->cols, sizeof(MPInt)*h_P0->nz, cudaMemcpyHostToDevice);

  printf("~~ after memcpy P0: %d -> %s\n", cudaGetLastError(),
    cudaGetErrorName(cudaGetLastError()));

  /* initialization of P1 */
  cudaMemset(P1.d_row_pointers, 0, sizeof(MPInt)*(h_P0->m+1));
  cudaMemset(P1.d_cols, 0, sizeof(MPInt)*h_P0->nz*2);
  printf("~~ after init: %d -> %s\n", cudaGetLastError(),
    cudaGetErrorName(cudaGetLastError()));

  printf("m (init): %d\n", m);  /* initial dimension size */

  for (p = 0; p < n_levels; ++p)
  {
    MPInt n_threads_total = (m+blk-1)/blk;

    dim3 threads_per_block(n_threads, 1, 1);
    dim3 blocks_per_grid;
    blocks_per_grid.x = (n_threads_total+n_threads-1)/n_threads;
    blocks_per_grid.y = 1;
    blocks_per_grid.z = 1;

    //dim3 threads_per_block_c(n_threads, 1, 1);
    //dim3 blocks_per_grid_c;

    P1.m = (m+blk-1)/blk;

  int *te = (MPInt*)mp_malloc(sizeof(int)*m);
  cudaMemcpy(te, &P0.d_row_pointers[1], sizeof(MPInt)*m, cudaMemcpyDeviceToHost);
  for (int z = 0; z < 20; ++z)
  {
    printf("te[%d]: %d, h_P0->rows_end[%d]: %d\n", z, te[z], z, h_P0->rows_end[z]);
  }

    /* computes number of nonzeros of each row of P1 */
    mp_cuda_blocking_contract_2_symbolic<<<blocks_per_grid, threads_per_block>>>
    (
      blk,
      m,
      P0.d_row_pointers,
      P0.d_cols,
      P1.d_row_pointers,
      P1.d_cols,
      temp_array
    );

    printf("~~ P: %d, after contract: %d -> %s\n", p, cudaGetLastError(),
      cudaGetErrorName(cudaGetLastError()));

    MPInt *d_al = (MPInt*)mp_malloc(sizeof(MPInt)*10);
    cudaMemcpy(d_al, P1.d_cols, sizeof(MPInt)*10, cudaMemcpyDeviceToHost);
    printf("d_al[0]: %d\n", d_al[0]);
    printf("d_al[1]: %d\n", d_al[1]);
    printf("d_al[2]: %d\n", d_al[2]);

    P0.m = (m+blk-1)/blk;

    printf("synchronizing...\n");
    cudaDeviceSynchronize();

    /* computes P1.d_row_pointers (partial sums) using prefix_scan */
    cudaMemcpy((void*)temp_array, P1.d_row_pointers, sizeof(MPInt)*(P1.m+1), cudaMemcpyDeviceToDevice);
    cudaMemset((void*)&P1.d_row_pointers[0], 0, sizeof(MPInt));
    mp_cuda_d_parsum(n_threads, P1.m, temp_array, &P1.d_row_pointers[1],
      d_acc);
    cudaDeviceSynchronize();

    cudaMemcpy(&P1.nz, &P1.d_row_pointers[P1.m], sizeof(MPInt), cudaMemcpyDeviceToHost);
    printf("P1.nz: %d\n", P1.nz);
    mp_cuda_blocking_contract_2<<<blocks_per_grid, threads_per_block>>>
    (
      blk,
      m,
      P0.d_row_pointers,
      P0.d_cols,
      P1.d_row_pointers,
      P1.d_cols,
      temp_array
    );

    cudaMemcpy(te, &P0.d_row_pointers[1], sizeof(MPInt)*m, cudaMemcpyDeviceToHost);
    for (int z = 0; z < 40; ++z)
    {
      printf("te[%d]: %d, h_P0->rows_end[%d]: %d\n", z, te[z], z, h_P0->rows_end[z]);
    }
    mp_free(te);


//    cudaDeviceSynchronize();
    int t = 0;
    int t1 = 0;
    int t2 = 0;
    int t3 = 0;
//    cudaMemcpy(&t, &P0.d_row_pointers[0], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t1, &P0.d_row_pointers[1], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t2, &P0.d_row_pointers[2], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t3, &P0.d_row_pointers[3], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    printf("P0.nz: %d, t: %d, t1: %d, t2: %d, t3: %d\n", P0.nz, t, t1, t2, t3);
//    cudaMemcpy(&t, &P1.d_row_pointers[0], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t1, &P1.d_row_pointers[1], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t2, &P1.d_row_pointers[2], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t3, &P1.d_row_pointers[3], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    printf("P1.nz: %d, t: %d, t1: %d, t2: %d, t3: %d\n", P1.nz, t, t1, t2, t3);
//
//    printf("after\n");
//
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(&P1.nz, &P1.d_row_pointers[P1.m], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    printf("m: %d, P1.m: %d, P1.nz: %d\n", m, P1.m-1, P1.nz);
//    printf("~~ P: %d, after prefix_scan: %d -> %s\n", p, cudaGetLastError(),
//      cudaGetErrorName(cudaGetLastError()));
//
//    cudaDeviceSynchronize();
//    printf("~~ P: %d, after compact: %d -> %s\n", p, cudaGetLastError(),
//      cudaGetErrorName(cudaGetLastError()));
//
//
//    cudaMemcpy(&t, &P1.d_row_pointers[0], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t1, &P1.d_row_pointers[1], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t2, &P1.d_row_pointers[2], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&t3, &P1.d_row_pointers[3], sizeof(MPInt), cudaMemcpyDeviceToHost);
//    printf("P1.nz: %d, t: %d, t1: %d, t2: %d, t3: %d\n", P1.nz, t, t1, t2, t3);
//

    mp_cuda_psy_expand_distance_22_symbolic<<<blocks_per_grid, threads_per_block>>>
    (
      P1.m,
      P1.nz,
      P0.m,
      P0.nz,
      P1.d_row_pointers,
      P1.d_cols,
      P0.d_row_pointers,
      P0.d_cols,
      temp_array
    );

    cudaDeviceSynchronize();
    printf("~~ P: %d, after symbolic expand: %d -> %s\n", p, cudaGetLastError(),
      cudaGetErrorName(cudaGetLastError()));

    cudaMemcpy((void*)temp_array, &P0.d_row_pointers[1], sizeof(MPInt)*(P0.m), cudaMemcpyDeviceToDevice);
    cudaMemset((void*)&P0.d_row_pointers[0], 0, sizeof(MPInt));

    mp_cuda_d_parsum(n_threads, P0.m, temp_array, &P0.d_row_pointers[1], d_acc);
    cudaMemcpy(&P0.nz, &P0.d_row_pointers[P0.m], sizeof(MPInt), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t, &P0.d_row_pointers[0], sizeof(MPInt), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t1, &P0.d_row_pointers[1], sizeof(MPInt), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t2, &P0.d_row_pointers[2], sizeof(MPInt), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t3, &P0.d_row_pointers[3], sizeof(MPInt), cudaMemcpyDeviceToHost);
    printf("P0.nz: %d, t: %d, t1: %d, t2: %d, t3: %d\n", P0.nz, t, t1, t2, t3);
    cudaDeviceSynchronize();

//    /* computes the number of nonzeros that are generated via pattern expansion */
    int N = 30;
    int *t_vec = (int*)mp_malloc(sizeof(MPInt)*N);
    //cudaMemcpy((void*)t_vec, (void*)P0.d_row_pointers, sizeof(MPInt)*10, cudaMemcpyDeviceToHost);
    //cudaMemcpy((void*)t_vec, (void*)P1.d_row_pointers, sizeof(MPInt)*N, cudaMemcpyDeviceToHost);
    //printf("[P1] before\n");
    //for (int k = 0; k < N; ++k)
    //{
    //  printf("t_vec[%d]: %d\n", k, t_vec[k]);
    //}
    cudaMemcpy((void*)t_vec, (void*)P0.d_row_pointers, sizeof(MPInt)*N, cudaMemcpyDeviceToHost);
    printf("[P0] before\n");
    for (int k = 0; k < N; ++k)
    {
      printf("t_vec[%d]: %d\n", k, t_vec[k]);
    }
    //cudaMemcpy((void*)t_vec, (void*)P1.d_cols, sizeof(MPInt)*N, cudaMemcpyDeviceToHost);
    //printf("[P1] before\n");
    //for (int k = 0; k < N; ++k)
    //{
    //  printf("t_vec[%d]: %d (cols)\n", k, t_vec[k]);
    //}
    //mp_free((void*)t_vec);


    printf("~~ P: %d, after second parsum: %d -> %s\n", p, cudaGetLastError(),
      cudaGetErrorName(cudaGetLastError()));

    mp_cuda_psy_expand_distance_22<<<blocks_per_grid, threads_per_block>>>
    (
      P1.m,
      P1.nz,
      P0.m,
      &P0.nz,
      P1.d_row_pointers,
      P1.d_cols,
      P0.d_row_pointers,
      P0.d_cols,
      temp_array
    );

    printf("~~ P: %d, after expand: %d -> %s\n", p, cudaGetLastError(),
      cudaGetErrorName(cudaGetLastError()));

    m = (m+blk-1)/blk;
    cudaDeviceSynchronize();
  }

  printf("-->m: %d\n", m);
  h_P0->m = P0.m;
  h_P0->nz = P0.nz;
  h_P1->m = P1.m;
  h_P1->nz = P1.nz;

  /* transfers back pattern entries from device to host */
  cudaMemcpy(h_P0->rows_start, P0.d_row_pointers, sizeof(MPInt)*P0.m, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_P0->rows_end, &P0.d_row_pointers[1], sizeof(MPInt)*P0.m, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_P0->cols, P0.d_cols, sizeof(MPInt)*P0.nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_P1->rows_start, P1.d_row_pointers, sizeof(MPInt)*P1.m, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_P1->rows_end, &P1.d_row_pointers[1], sizeof(MPInt)*P1.m, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_P1->cols, P1.d_cols, sizeof(MPInt)*P1.nz, cudaMemcpyDeviceToHost);
  h_P0->nz = P0.nz;
  h_P0->m = P0.m;

  h_P1->nz = P1.nz;
  h_P1->m = P1.m;

  printf("h_P1->cols[0]: %d, h_P1->cols[1]: %d, h_P1->cols[2]: %d\n", h_P1->cols[0], h_P1->cols[1], h_P1->cols[2]);
  printf("h_P0->cols[0]: %d, h_P0->cols[1]: %d, h_P0->cols[2]: %d\n", h_P0->cols[0], h_P0->cols[1], h_P0->cols[2]);
  printf("h_P0->nz: %d\n", h_P0->nz);
  for (int i1 = 0; i1 < 30; ++i1)
  {
    printf("%d\n", h_P0->cols[i1]);
  }

  /* sets patterns for entry selection */
  context->P_prev = h_P1;
  context->P = h_P0;
  context->m_P = P0.m;

  /* frees dynamically allocated device memory */
  cudaFree(P0.d_row_pointers);
  cudaFree(P0.d_cols);
  cudaFree(P1.d_row_pointers);
  cudaFree(P1.d_cols);
  cudaFree(temp_array);
  cudaFree(d_acc);


}

__global__ void mp_cuda_blocking_contract_2_symbolic /* without swapping in memory (not done already) */
(
  MPInt blk,                  /* block size */
  MPInt m,
  MPInt *P0_d_row_ptr,
  MPInt *P0_d_cols,
  MPInt *P1_d_row_ptr,
  MPInt *P1_d_cols,
  MPInt *temp_array          /* temporary storage array */
)
{
  MPInt i = (blockIdx.x*blockDim.x + threadIdx.x)*blk;
  MPInt j = 0;  /* parses across m_new (blocks) */

  MPInt c_P1 = -1; /* holds block column of P1 */
  MPInt nz = 0;    /* counts nonzeros of P1 */

  /* parses rows of P0 in strides of blk (block_size) */
  MPInt min = m;        /* minimum value, is set to max possible value = P1->m */
  MPInt jmin = -1;      /* index of temp_array */
  MPInt c_prev = -1;    /* previous column encountered */
  MPInt new_entry = 0;  /* marks a new entry */
  MPInt term = 1;       /* used to terminate while loop */
  MPInt nz_true = 0;

  /* copies P0->rows_start to temp_array */
  //if (i == 159)
  //if (i < 157)
  //if (i < 158)
  //if (i == 128) // problem found via count
  //if (i < m)
  //if (i < 1)
  {
    for (j = 0; j < blk; ++j)
    {
      temp_array[i+j] = P0_d_row_ptr[i+j];
    }

    /* merges blk rows at a time */
    while (term)
    {
      /* parses entries from consecutive rows */
      for (j = 0; j < blk; ++j)
      {
        /* if not reached the end of the row */
        if (temp_array[i+j] < P0_d_row_ptr[i+j+1])
        {
          c_P1 = P0_d_cols[temp_array[i+j]]/blk; /* computes P1 column */
          //printf("i: %d, j: %d, P0_d_cols[temp_array[%d]:%d]: %d/%d, c_P1: %d, %d, blk: %d\n", i, j, i+j, P0_d_cols[temp_array[i+j]], P0_d_cols[temp_array[i+j]]/blk, temp_array[i+j], c_P1, c_P1, blk);

          /* if column is the new min */
          while ((c_P1 == c_prev) && (temp_array[i+j] < P0_d_row_ptr[i+j+1]))
          {
            temp_array[i+j] += 1;
            c_P1 = P0_d_cols[temp_array[i+j]]/blk; /* computes P1 column */
          }

          //printf("i: %d, j: %d, P0_d_cols[temp_array[%d]]: %d, P0_d_cols[temp_array[%d]]/blk: %d, temp_array[i+j]: %d, c_P1: %d\n", i, j, i+j, P0_d_cols[temp_array[i+j]], i+j, P0_d_cols[temp_array[i+j]]/blk, temp_array[i+j], c_P1);

          if ((c_P1 < min) && (temp_array[i+j] < P0_d_row_ptr[i+j+1]))
          {
            min = c_P1;     /* minimum value of block column (P1) */
            jmin = j;       /* row of P0 that contains newly added entry */
            new_entry = 1;  /* marks new entry in P1 */
          }
        }
      }

      /* if a new block column is added to P1 */
      if (new_entry)
      {
        //P1_d_cols[P1_d_row_ptr[i/blk]+nz] = min;
        //if (P1_d_row_ptr[i/blk]+nz == 0)
        //{
        //  printf("i: %d\n", i);
        //}
        //if (i == 0)
        //{
        //  printf("blockIdx.x: %d, P1_d_row_ptr[i/blk]+nz: %d, P1_d_cols[P1_d_row_ptr[i/blk]+nz]: %d, min: %d\n", blockIdx.x, P1_d_row_ptr[i/blk]+nz, P1_d_cols[P1_d_row_ptr[i/blk]+nz], min);
        //  //printf("i/blk: %d> %d, min: %d, P1_d_row_ptr[i/blk]+nz: %d\n", i/blk, P1_d_cols[P1_d_row_ptr[i/blk]+nz], min, P1_d_row_ptr[i/blk]+nz);
        //}
        //if (c_P1 != c_prev)
        //{
        //  nz_true += 1;  /* counts one more nonzero entry for P1 */
          //printf("  -> c_prev: %d, c_P1: %d\n", c_prev, c_P1);
        //}
        nz += 1;
        //printf("  >out, temp_array[0]: %d -> %d, temp_array[1]: %d -> %d, nz: %d, nz_true: %d, c_prev: %d\n", temp_array[0], P0_d_cols[temp_array[0]], temp_array[1], P0_d_cols[temp_array[1]], nz, nz_true, c_prev);
        c_prev = min;
        new_entry = 0;
        min = m;
        temp_array[i+jmin] += 1;
      }
      else
      {
        //printf("  >out, temp_array[0]: %d -> %d, temp_array[1]: %d -> %d, nz: %d, nz_true: %d, c_prev: %d\n", temp_array[0], P0_d_cols[temp_array[0]], temp_array[1], P0_d_cols[temp_array[1]], nz, nz_true, c_prev);
      }

      /* recompute term */
      term = 0;
      for (j = 0; j < blk; ++j)
      {
        term += (P0_d_row_ptr[i+j+1]-temp_array[i+j]);
      }
    }

    /* updates row endpoints, and total number of nonzero entryes */
    P1_d_row_ptr[(blockIdx.x*blockDim.x + threadIdx.x)] = nz;
    //P1_d_row_ptr[(blockIdx.x*blockDim.x + threadIdx.x)] = nz_true;
    //if (nz < 10)
    //{
      //printf("  >>>>(*)>(*)> nz: %d, nz_true: %d, p0: %d\n", nz, nz_true, P0_d_row_ptr[i+j+1]-P0_d_row_ptr[i+j]);
    //}
  }
}

__global__ void mp_cuda_blocking_contract_2 /* without swapping in memory (not done already) */
(
  MPInt blk,                  /* block size */
  MPInt m,
  MPInt *P0_d_row_ptr,
  MPInt *P0_d_cols,
  MPInt *P1_d_row_ptr,
  MPInt *P1_d_cols,
  MPInt *temp_array          /* temporary storage array */
)
{
  MPInt i = (blockIdx.x*blockDim.x + threadIdx.x)*blk;
  MPInt j = 0;  /* parses across m_new (blocks) */

  MPInt c_P1 = -1; /* holds block column of P1 */
  MPInt nz = 0;    /* counts nonzeros of P1 */

  /* parses rows of P0 in strides of blk (block_size) */
  MPInt min = m;        /* minimum value, is set to max possible value = P1->m */
  MPInt jmin = -1;      /* index of temp_array */
  MPInt c_prev = -1;    /* previous column encountered */
  MPInt new_entry = 0;  /* marks a new entry */
  MPInt term = 1;       /* used to terminate while loop */

  /* copies P0->rows_start to temp_array */
  if (i < m)
  {
    for (j = 0; j < blk; ++j)
    {
      temp_array[i+j] = P0_d_row_ptr[i+j];
    }

    /* merges blk rows at a time */
    while (term)
    {
      /* parses entries from consecutive rows */
      for (j = 0; j < blk; ++j)
      {
        /* if not reached the end of the row */
        if (temp_array[i+j] < P0_d_row_ptr[i+j+1])
        {
          c_P1 = P0_d_cols[temp_array[i+j]]/blk; /* computes P1 column */

          /* if column is the new min */
          while ((c_P1 == c_prev) && (temp_array[i+j] < P0_d_row_ptr[i+j+1]))
          {
            temp_array[i+j] += 1;
            c_P1 = P0_d_cols[temp_array[i+j]]/blk; /* computes P1 column */
          }

          if ((c_P1 < min) && (temp_array[i+j] < P0_d_row_ptr[i+j+1]))
          {
            min = c_P1;     /* minimum value of block column (P1) */
            jmin = j;       /* row of P0 that contains newly added entry */
            new_entry = 1;  /* marks new entry in P1 */
          }
        }
      }

      /* if a new block column is added to P1 */
      if (new_entry)
      {
        P1_d_cols[P1_d_row_ptr[i/blk]+nz] = min;
        nz += 1;  /* counts one more nonzero entry for P1 */
        c_prev = min;
        new_entry = 0;
        min = m;
        temp_array[i+jmin] += 1;
      }

      /* recompute term */
      term = 0;
      for (j = 0; j < blk; ++j)
      {
        term += (P0_d_row_ptr[i+j+1]-temp_array[i+j]);
      }
    }
  }

  /* updates row endpoints, and total number of nonzero entryes */
  //P1_d_row_ptr[(blockIdx.x*blockDim.x + threadIdx.x)] = nz;
}

__global__ void mp_cuda_blocking_compact
(
  MPInt n,
  MPInt blk,
  MPInt *d_rows_ptr_in,
  MPInt *d_rows_ptr_out,
  MPInt *d_cols
)
{
//  MPInt i = blockDim.x*blockIdx.x + threadIdx.x;
//  MPInt j = 0;
//  MPInt current = d_rows_ptr[i];
//  MPInt c_prev = -1;
//  blk = (1-(i+1)/n)*blk + ((i+1)/n)*(n%blk)
//
//  for (k = 0; k < blk; ++k)
//  {
//    for (j = d_rows_ptr[i+k]; j < d_rows_ptr[i+k+1]; ++j)
//    {
//      MPInt cond = (d_cols[j] > c_prev);
//      d_cols[current] = cond*d_cols[j] + (1-cond)*d_cols[current];
//      c_prev = cond*d_cols[current] + (1-cond)*c_prev;
//      current = cond*(current+1) + (1-cond)*current;
//    }
//  }
}

//__global__ void mp_cuda_psy_expand_distance_22_symbolic
//(
//  MPInt P0_m,           /* number of rows of P0 */
//  MPInt P0_nz,          /* number of nonzeros of P0 */
//  MPInt P1_m,           /* number of rows of P1 */
//  MPInt P1_nz,          /* number of nonzeros of P1 */
//  MPInt *P0_row_ptr,
//  MPInt *P0_cols,
//  MPInt *P1_row_ptr,
//  MPInt *P1_cols,
//  MPInt *temp_array
//)
//{
//  extern __shared__ int temp_shared[];
//
//  MPInt i = blockIdx.x*blockDim.x + threadIdx.x;
//  MPInt j = 0; /* nodes at distance-1 */
//  MPInt k = 0; /* nodes at distance-2 of v in V(A_csr) */
//
//  MPInt count = 0;
//  MPInt min = 0;
//  MPInt MAX_NODE = P0_m;
//  MPInt do_termination = 0;
//
//  MPInt c0 = 0;
//  MPInt c1 = 0;
//  MPInt c_select = 0;
//  MPInt nz = 0;
//
//  MPInt row_start = P0_row_ptr[i];
//  MPInt row_end = P0_row_ptr[i+1];
//
//  MPInt cond0 = 0;
//  MPInt cond1 = 0;
//  MPInt c_prev = -1;
//
//  MPInt j_select = 0;
//  //MPInt row_start_temp = P0_row_ptr[i];
//
//   MPInt count_debug = 0;
//
//  //if (i < 1)
//  //{
//    //printf("P0_row_ptr[0]: %d\n", P0_row_ptr[0]);
//   // P0_row_ptr[0] = 1;
//  //if (i < 1)
//  //if (i < 2)
//  {
//    if (row_start < row_end)
//    {
//      for (j = row_start; j < row_end; ++j)
//      {
//        c0 = P0_cols[j]; /* distance-1 neighbor */
//        temp_array[j] = P0_row_ptr[c0];
//        if (i == 0)
//        {
//          printf("temp_array[%d]: %d\n", j, temp_array[j]);
//        }
//      }
//
//      do
//      {
//        min = MAX_NODE;
//        do_termination = 0;
//        count = 0;
//
//        for (j = row_start; j < row_end; ++j)   /* parse distance-1 neighbors */
//        {
//          c0 = P0_cols[j]; /* distance-1 neighbor */
//          k = P0_row_ptr[c0];
//
//          cond0 = (k < P0_row_ptr[c0+1]);
//          cond1 = (c1 <= min);
//
//          c1 = cond0*P0_cols[k] + (1-cond0)*c1;
//          min = (cond0*cond1)*c1 + (1-cond0*cond1)*min;
//          //c_select = (cond0*cond1)*c0 + (1-cond0*cond1)*c_select;
//          j_select = (cond0*cond1)*j + (1-cond0*cond1)*j_select;
//          count = (cond0*cond1)*(count+1) + (1-cond0*cond1)*count;
//          if (i == 0)
//          {
//            printf("c1: %d, P0_cols[k]: %d, k: %d, c0: %d\n", c1, P0_cols[k], k, c0);
//          }
//
//          // lags here
//          do_termination = cond0*(do_termination+(P0_row_ptr[c0+1]-temp_array[j])) + (1-cond0)*do_termination;
//          //if (i == 0)
//          //{
//          //  printf("do_termination: %d, P0_row_ptr[c0+1]: %d, temp_array[j]: %d, P0_row_ptr[c0]: %d, j_select: %d, min: %d, c1: %d, k: %d -> %d, nz: %d\n", do_termination, P0_row_ptr[c0+1], temp_array[j], P0_row_ptr[c0], j_select, min, c1, k, P0_cols[k], nz);
//          //}
//        }
//        MPInt cond = (count > 0)&&((nz == 0) || ((nz > 0) && (c_prev != min)));  // error here in indexing
//        c_prev = min;
//        min = MAX_NODE;
//        if (i == 0)
//        {
//          printf("cond: %d, P1_cols[nz-1]: %d, c_prev: %d, min: %d\n", cond, P1_cols[nz-1], c_prev, min);
//        }
//        nz = cond*(nz+1) + (1-cond)*nz;
//        int ta = temp_array[j_select];
//        temp_array[j_select] = (count > 0)*(temp_array[j_select]+1) + (1-(count > 0))*temp_array[j_select];
//        count_debug += 1;
//        //printf("j_select: %d, ta: %d, temp_array[j_select]: %d, count: %d\n", j_select, ta, temp_array[j_select], count);
//        //printf("\n");
//
//        //if (do_termination > 0)
//        //{
//        //  printf("thread_id: %d, do_termination: %d\n", i, do_termination);
//        //}
//      } while((do_termination != 0));  //&& (count_debug < 10)
//    }
//    P1_row_ptr[i+1] = nz;
//    if (i == 0)
//    {
//      printf("______________> i: %d, row_start: %d, row_end: %d, nz: %d\n", i, row_start, row_end, nz);
//    }
//    //printf("i: %d, nz: %d\n", i, nz);
//    //if (i == P0_m-1)
//    //{
//    //  printf("      __________________> nz: %d\n", nz);
//    //}
//  }
//  //}
//  //printf("|>>out\n");
//}

__global__ void mp_cuda_psy_expand_distance_22_symbolic
(
  MPInt P0_m,           /* number of rows of P0 */
  MPInt P0_nz,          /* number of nonzeros of P0 */
  MPInt P1_m,           /* number of rows of P1 */
  MPInt P1_nz,          /* number of nonzeros of P1 */
  MPInt *P0_row_ptr,
  MPInt *P0_cols,
  MPInt *P1_row_ptr,
  MPInt *P1_cols,
  MPInt *temp_array
)
{
  MPInt i = blockIdx.x*blockDim.x + threadIdx.x;
  MPInt j = 0; /* nodes at distance-1 */
  MPInt k = 0; /* nodes at distance-2 of v in V(A_csr) */
  MPInt row_start = 0;
  MPInt row_end = 0;
  MPInt count = 0;
  MPInt min = 0;
  MPInt MAX_NODE = P0_m;
  MPInt do_termination = 0;
  MPInt c0 = 0;
  MPInt c1 = 0;
  MPInt c_select = 0;
  MPInt jmin = 0;
  MPInt nz = 0;
  MPInt c_prev = -1;

  row_start = P0_row_ptr[i];
  row_end = P0_row_ptr[i+1];
  if (i < P0_m)
  {
    if (row_start < row_end)
    {
      for (j = row_start; j < row_end; ++j)
      {
        c0 = P0_cols[j];
        temp_array[j] = P0_row_ptr[c0];
      }

      do
      {
        min = MAX_NODE;      /* maximum node id */
        do_termination = 0;  /* termination condition */
        count = 0;
        for (j = row_start; j < row_end; ++j)   /* parse distance-1 neighbors */
        {
          /* this should probably be replaced by a more efficient search, like
             an interpolation search tree but for the moment is ok. */

          c0 = P0_cols[j];           /* distance-1 neighbor */
          k = temp_array[j];
          if (k < P0_row_ptr[c0+1])   /* check if row of P1 is not empty */
          {
            c1 = P0_cols[k];         /* access column of P1 */
            if (c1 <= min)
            {
              min = c1;
              jmin = j;
              count++;
            }
            do_termination += (P0_row_ptr[c0+1]-temp_array[j]);
          }
        }

        if (count > 0) /* there is at least one node */
        {
          if ((nz == 0) || ((nz > 0) && (c_prev != min)))
          {
            nz += 1;
            c_prev = min;
            min = MAX_NODE;
          }
          temp_array[jmin] += 1;
        }
      } while(do_termination != 0);

      P1_row_ptr[i+1] = nz;
    }
  }
}

__global__ void mp_cuda_psy_expand_distance_22
(
  MPInt P0_m,
  MPInt P0_nz,
  MPInt P1_m,
  MPInt *P1_nz,
  MPInt *P0_row_ptr,
  MPInt *P0_cols,
  MPInt *P1_row_ptr,
  MPInt *P1_cols,
  MPInt *temp_array
)
{
  MPInt i = blockIdx.x*blockDim.x + threadIdx.x;
  MPInt j = 0; /* nodes at distance-1 */
  MPInt k = 0; /* nodes at distance-2 of v in V(A_csr) */
  MPInt row_start = 0;
  MPInt row_end = 0;
  MPInt count = 0;
  MPInt min = 0;
  MPInt MAX_NODE = P0_m;
  MPInt do_termination = 0;
  MPInt c0 = 0;
  MPInt c1 = 0;
  MPInt c_select = 0;
  MPInt nz = 0;
  MPInt jmin = -1;
  MPInt c_prev = -1;

  MPInt count_debug = 0;

  row_start = P0_row_ptr[i];
  row_end = P0_row_ptr[i+1];

  if (i < P0_m)
  {
    if (row_start < row_end)
    {
      for (j = row_start; j < row_end; ++j)
      {
        c0 = P0_cols[j];
        temp_array[j] = P0_row_ptr[c0];
      }

      nz = P1_row_ptr[i];
      do
      {
        min = MAX_NODE;      /* maximum node id */
        do_termination = 0;  /* termination condition */
        count = 0;

        for (j = row_start; j < row_end; ++j)   /* parse distance-1 neighbors */
        {
          /* this should probably be replaced by a more efficient search, like
             an interpolation search tree but for the moment is ok. */

          c0 = P0_cols[j];           /* distance-1 neighbor */
          k = temp_array[j];
          if (k < P0_row_ptr[c0+1])   /* check if row of P1 is not empty */
          {
            c1 = P0_cols[k];         /* access column of P1 */
            if (c1 <= min)
            {
              min = c1;
              c_select = c0;
              jmin = j;
              count++;
            }
            do_termination += (P0_row_ptr[c0+1]-temp_array[j]);
          }
        }

        if (count > 0) /* there is at least one node */
        {
          if ((nz == 0) || ((nz > 0) && (c_prev != min)))
          {
            P1_cols[nz] = min;
            c_prev = min;
            nz += 1;
          }
          temp_array[jmin] += 1;
        }
      } while(do_termination != 0);
    }
  }
}

//__global__ void mp_cuda_psy_expand_distance_22
//(
//  MPInt P0_m,
//  MPInt P0_nz,
//  MPInt P1_m,
//  MPInt *P1_nz,
//  MPInt *P0_row_ptr,
//  MPInt *P0_cols,
//  MPInt *P1_row_ptr,
//  MPInt *P1_cols
//)
//{
//  MPInt i = blockIdx.x*blockDim.x + threadIdx.x;  /* row index of P0 */
//  MPInt j = 0; /* nodes at distance-1 */
//  MPInt k = 0; /* nodes at distance-2 of v in V(A_csr) */
//
//  MPInt row_start = 0;
//  MPInt row_end = 0;
//
//  MPInt count = 0;
//  MPInt min = 0;
//  MPInt MAX_NODE = P0_m;
//  MPInt do_termination = 0;
//
//  MPInt c0 = 0;
//  MPInt c1 = 0;
//  MPInt c_select = 0;
//
//  P1_row_ptr[0] = 0;
//  P1_row_ptr[1] = 0;
//  *P1_nz = 0;
//  P1_m = P0_m;
//
//  //mp_matrix_i_set(MP_COL_MAJOR, P0->m, 1, P1->cols, P0->m, 0);
//
//  P1_row_ptr[i] = *P1_nz;   /* sets output row_start */
//  row_start = P0_row_ptr[i];
//  row_end = P0_row_ptr[i+1];
//
//  if (row_start < row_end)
//  {
//    do
//    {
//      min = MAX_NODE;      /* maximum node id */
//      do_termination = 0;  /* termination condition */
//      count = 0;
//
//      for (j = row_start; j < row_end; ++j)   /* parse distance-1 neighbors */
//      {
//        c0 = P0_cols[j];           /* distance-1 neighbor */
//        k = P0_row_ptr[c0];
//
//        MPInt cond0 = (k < P0_row_ptr[c0]);
//        MPInt cond1 = (c1 <= min);
//
//        c1 = (cond0)*P0_cols[k] + (1-cond0)*c1;
//        min = (cond0*cond1)*c1 + (1-cond0*cond1)*min;
//        c_select = (cond0*cond1)*c0 + (1-cond0*cond1)*c_select;
//        count = (cond0*cond1)*(count+1) + (1-cond0*cond1)*count;
//        do_termination = (cond0)*(do_termination+P0_row_ptr[c0+1]-P0_row_ptr[c0])
//                       + (1-cond0)*do_termination;
//      }
//
//      MPInt cond = (count > 0)*((P1_nz == 0) || ((P1_nz > 0) && (P1_cols[*P1_nz-1] != min)));
//      P1_cols[*P1_nz] = (cond)*min + (1-cond)*(P1_cols[*P1_nz]);
//      *P1_nz = (cond)*(*P1_nz+1) + (1-cond)*(*P1_nz);
//      P0_row_ptr[c_select] = (count>0)*(P0_row_ptr[c_select]+1) + (1-count>0)*P0_row_ptr[c_select];
//    } while(do_termination != 0);
//
//    //for (j = row_start; j < row_end; ++j)
//    //{
//    //  k = P0->cols[j];
//    //  if (k > 0)
//    //  {
//    //    P0->rows_start[k] = P0->rows_end[k-1];
//    //  }
//    //  else if (k == 0)
//    //  {
//    //    P0->rows_start[0] = 0;
//    //  }
//    //}
//  }
//  P1_row_ptr[i+1] = *P1_nz;
//}


//void mp_cuda_blocking_psy_fA_2 /* using mp_blocking_contract_2 () */
//(
//  MPContext *context
//)
//{
//  /* unpacking context input */
//  MPInt p = 0;
//  MPInt blk = context->probing.blocking.blk;
//  MPInt n_levels = context->n_levels;
//
//  MPInt *temp_array = context->memory_probing;
//  MPInt *temp_inverted_array = &temp_array[context->m_P/blk];
//
//  context->m_P = context->m_A;
//  MPInt m_P = (context->m_A+blk-1)/blk;
//
//  MPPatternCsr *P0 = &context->pattern_array.blocking[0];
//  MPPatternCsr *P1 = &context->pattern_array.blocking[1];
//
//  P0->m = context->m_A;
//  P0->nz = context->nz_A;
//  P1->m = context->m_A;
//  P1->nz = context->nz_A;
//
//  /* coarsens and expands to approximate S(A^{-1}) */
//  for (p = 0; p < n_levels; ++p)
//  {
//    /* reset test_array and temp_inverted_array */
//    mp_matrix_i_set(MP_COL_MAJOR, m_P, 1, temp_array, m_P, 0);
//    mp_matrix_i_set(MP_COL_MAJOR, m_P, 1, temp_inverted_array, m_P, -1);
//
//    P1->rows_start[0] = 0;
//    P1->rows_end[0] = 0;
//
//    /* contract and expand */
//    mp_blocking_contract_2(blk, P0, P1, temp_array, temp_inverted_array);
//    mp_psy_expand_distance_22_new(context->memory_increment, P1, P0);
//  }
//
//  context->P_prev = P1;
//  context->P = P0;
//}

//__global__ void mp_cuda_blocking_contract
//(
//  int n_levels,
//  int blk,
//  int m,
//  int nz,
//  int *d_row_pointers_A,
//  int *d_cols_A,
//  int *d_row_pointers_B,
//  int *d_cols_B,
//  int *temp_array,
//  MPPatternHeap_Cuda *T
//)
//{
//  extern __shared__ int mem_sh[];
//  int i = 0;
//  int thread_id = threadIdx.x;
//
//  for (i = 0; i < n_levels; ++i)
//  {
//    __syncthreads();  /* wait for threads of the block */
//
//    /*-------------------*/
//    /* PHASE 1: contract */
//    /*-------------------*/
//
//    d_cols_A[thread_id] = d_cols_A[thread_id]/blk;
//
//    /*---------------------*/
//    /* PHASE 2: prefix sum */
//    /*---------------------*/
//
//    if (thread_id < (m+blk-1)/blk)
//    {
//      int j = 0;
//      for (i = 0; i < blk; ++i)
//      {
//        temp_array[i] = d_row_pointers[thread_id+i];
//      }
//
//      int term = 1;
//      int c_prev = 0;
//      while (term)
//      {
//        term = 0;
//        for (i = 0; i < blk; ++i)
//        {
//          j = temp_array[i];
//          /* ATTENTION: have to remove duplicates at delete! (create new function)  */
//          if (d_cols[j] != c_prev)
//          {
//            mp_heap_min_cuda_fibonacci_insert(&T->d_heap_array[thread_id], d_cols[j]);
//            d_row_pointers_o[thread_id] += 1;
//          }
//          c_prev = d_cols[j];
//          temp_array[i] += 1;
//          term += (d_row_pointers[thread_id+i+1]-temp_array[i]);
//        }
//      }
//    }
//
//
//    /*--------------------------------------*/
//    /* PHASE 3: sort d_cols_A into d_cols_B */
//    /*--------------------------------------*/
//
//    if (thread_id < (m+blk-1)/blk)
//    {
//      int j = 0;
//      for (i = 0; i < blk; ++i)
//      {
//        temp_array[i] = d_row_pointers[thread_id+i];
//      }
//
//      int term = 1; /* terminates merging */
//      int c_prev = 0;
//      while (term)  /* merging phase */
//      {
//        term = 0;
//        for (i = 0; i < blk; ++i)
//        {
//          j = temp_array[i];
//          /* ATTENTION: have to remove duplicates at delete! (create new function)  */
//          if (d_cols[j] != c_prev)
//          {
//            mp_heap_min_cuda_fibonacci_insert(&T->d_heap_array[thread_id], d_cols[j]);
//            d_row_pointers_o[thread_id] += 1;
//          }
//          c_prev = d_cols[j];
//          temp_array[i] += 1;
//          term += (d_row_pointers[thread_id+i+1]-temp_array[i]);
//        }
//
//        for (i = 0; i < blk; ++i)
//        {
//          while
//          (
//            (temp_array[i] <= min) &&
//            (temp_array[i] < d_row_pointers_A[i+1])
//          )
//          {
//            temp_array[i] += 1;
//          }
//        }
//      }
//    }
//
//    /*---------------------*/
//    /* PHASE 3: prefix sum */
//    /*---------------------*/
//
//    if (thread_id < (m+blk-1)/blk)
//    {
//      //prefix_scan_device
//      //(
//      //  thread_id,
//      //  m,            /* number of entries */
//      //  d_cols,       /* input vector */
//      //  mem_sh
//      //)
//    };
//
//    /*---------------------------------*/
//    /* PHASE 4: compact rows of output */
//    /*---------------------------------*/
//    if (thread_id < m)
//    {
//      for (i = mem_sh[thread_id]; i < mem_sh[thread_id+1]; ++i)
//      {
//      }
//    }
//  }
//}

//void mp_cuda_blocking_psy_fA
//(
//  MPContext *context
//)
//{
//  //int m_B = 16042;
//  //int n_threads = 256;
//  //int n_blocks = (m_B+n_threads-1)/n_threads;
//
//  //MPPatternCsr *P0 = &context->pattern_array.blocking[0];
//  //MPPatternCsr *P1 = &context->pattern_array.blocking[1];
//
//  //dim3 n_threads_per_block;
//  //n_threads_per_block.x = N;
//  //n_threads_per_block.y = 1;
//  //n_threads_per_block.z = 1;
//
//  //mp_cuda_blocking_psy_fA_kernel<<<n_blocks, n_threads_per_block>>>(m_B, P0, P1);
//  //cudaDeviceSynchronize();
//}

__global__ void vec_add
(
  float *a,
  float *b,
  float *c
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

//int main(int argc, char **argv)
//{
//  int i = 0;
//  int n = 32;
//  float *v_in = (float*)malloc((sizeof *v_in)*n);
//  float *v_out = (float*)malloc((sizeof *v_in)*n);
//  float *d_v_in = (float*)malloc((sizeof *v_in)*n);
//  float *d_v_out = (float*)malloc((sizeof *v_in)*n);
//
//  cudaMalloc((void**)&d_v_in, (sizeof *d_v_in)*n);
//  cudaMalloc((void**)&d_v_out, (sizeof *d_v_out)*n);
//
//  /* input instance #1 */
//  //for (i = 0; i < n; ++i)
//  //{
//  //  v_in[i] = 1.0;
//  //}
//  //memset(d_v_out, (sizeof *d_v_out)*n, 0);
//
//  /* input instance #2 */
//  for (i = 0; i < n; ++i)
//  {
//    v_in[i] = i;
//  }
//
//  cudaError_t status;
//  status = cudaMemcpy((void*)d_v_in, (void*)v_in, (sizeof *d_v_in)*n, cudaMemcpyHostToDevice);
//  printf("(1) status: %d -> %s\n", (int)status, cudaGetErrorName(status));
//  cudaMemcpy(d_v_out, v_out, (sizeof *d_v_out)*n, cudaMemcpyHostToDevice);
//
//  dim3 cuda_blocks_per_grid(1, 1, 1);
//  dim3 cuda_threads_per_block(16, 1, 1);
//
//  prefix_scan<<<cuda_blocks_per_grid, cuda_threads_per_block, sizeof(int)*n>>>
//  (
//    n,        /* number of inputs left and right */
//    d_v_in,   /* input vector */
//    d_v_out   /* output vector */
//  );
//
//  cudaDeviceSynchronize();
//  status = cudaMemcpy(v_out, d_v_out, (sizeof *v_out)*n, cudaMemcpyDeviceToHost);
//  printf("(2) status: %d -> %s\n", (int)status, cudaGetErrorName(status));
//
//  for (i = 0; i < n; ++i)
//  {
//    printf("v_out[%d]: %f\n", i, v_out[i]);
//  }
//
//  free(v_out);
//  free(v_in);
//  cudaFree(d_v_in);
//  cudaFree(d_v_out);
//  return 0;
//}
