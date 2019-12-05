#include "omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

// Arrays size <= SMALL switches to insertion sort
/*int SMALL = 32;*/
/*#define C (262144/4*3)*/
int C = (32768);
int SMALL = 32;
#define TIME_ROUND 3
#define ELEMENT_MAX_RANGE 0x3f3f3f3f
/*#define MERGE_WAY 4*/
/*#define TEST_CACHE_INSTEAD*/

double get_time(void) {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (t.tv_sec + ((double)(t.tv_usec)) / 1e6);
}
void run_segmented_merge_path(int a[], long long int size, int temp[],
                              int threads, int parallel_merge);
void segmented_mergesort_merge_path(int a[], long long int size, int temp[],
                                    int threads);
void run_merge_path(int a[], long long int size, int temp[], int threads,
                    int parallel_merge);
void mergesort_merge_path(int a[], long long int size, int temp[], int threads);

// binary merge
int binsearch_findfirst_g(int *array, long long int n, int target);
int binsearch_findfirst_ge(int *array, long long int n, int target);
void merge_binary(int a[], long long int size, int temp[], int threads);
void insertion_sort(int a[], long long int size);
void mergesort_serial(int a[], long long int size, int temp[]);
void mergesort_parallel_omp(int a[], long long int size, int temp[],
                            int threads, int parallel_merge);
// merge path
void merge_path_parallel(int a[], long long int size, int temp[], int threads);
void segmented_merge_path_parallel(int a[], long long int size, int temp[],
                                   int threads);
void diagnal_intersection(int a[], int b[], long long int asize,
                          long long int bsize, int threads,
                          long long int *astart, long long int *bstart,
                          long long int i, long long int diag);

void merge_serial(int a[], long long int size, int temp[]);
void merge_serial_general(int a[], int b[], long long int astart,
                          long long int bstart, int temp[],
                          long long int tempstart, long long int size,
                          long long int asize, long long int bsize,
                          long long int *ainc, long long int *binc);
void run_omp(int a[], long long int size, int temp[], int threads,
             int parallel_merge);
void run_test(int a[], long long int size, int temp[], int threads,
              int parallel_merge, int tmp_refresh[],
              void (*run_omp)(int a[], long long int size, int temp[],
                              int threads, int parallel_merge),
              int iter_threads, int test_small);
int main(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  puts("-OpenMP Recursive Mergesort-\t");
  // Check arguments
  if (argc < 3) /* argc must be 3 for proper execution! */
  {
    printf("Usage: %s array-size max-number-of-threads\n"
           "\t[--merge-type-{plain or binary or mergepath or "
           "seg-mergepath}]\n\t[--inc-threads]\n\t[--test-small]\n",
           argv[0]);
    return 1;
  }
  // Get arguments
  long long int size = atoll(argv[1]); // Array size
  int threads = atoi(argv[2]);         // Requested number of threads
  int merge_type = 0;
  int inc_threads = 0;
  int test_small = 0;
  if ((argc > 3 &&
       strncmp("--merge-type", argv[3], strlen("--merge-type")) == 0)) {
    if (strcmp(argv[3] + strlen("--merge-type-"), "plain") == 0)
      merge_type = 1;
    else if (strcmp(argv[3] + strlen("--merge-type-"), "binary") == 0)
      merge_type = 2;
    else if (strcmp(argv[3] + strlen("--merge-type-"), "seg-mergepath") == 0)
      merge_type = 3;
    else if (strcmp(argv[3] + strlen("--merge-type-"), "mergepath") == 0)
      merge_type = 4;
  } else if (argc > 4 &&
             strncmp("--merge-type", argv[4], strlen("--merge-type")) == 0) {
    if (strcmp(argv[4] + strlen("--merge-type-"), "plain") == 0)
      merge_type = 1;
    else if (strcmp(argv[4] + strlen("--merge-type-"), "binary") == 0)
      merge_type = 2;
    else if (strcmp(argv[4] + strlen("--merge-type-"), "seg-mergepath") == 0)
      merge_type = 3;
    else if (strcmp(argv[4] + strlen("--merge-type-"), "mergepath") == 0)
      merge_type = 4;
  } else if (argc > 5 &&
             strncmp("--merge-type", argv[5], strlen("--merge-type")) == 0) {
    if (strcmp(argv[5] + strlen("--merge-type-"), "plain"))
      merge_type = 1;
    else if (strcmp(argv[5] + strlen("--merge-type-"), "binary") == 0)
      merge_type = 2;
    else if (strcmp(argv[5] + strlen("--merge-type-"), "seg-mergepath") == 0)
      merge_type = 3;
    else if (strcmp(argv[5] + strlen("--merge-type-"), "mergepath") == 0)
      merge_type = 4;
  }
  if ((argc > 3 && strcmp("--inc-threads", argv[3]) == 0) ||
      (argc > 4 && strcmp("--inc-threads", argv[4]) == 0) ||
      (argc > 5 && strcmp("--inc-threads", argv[5]) == 0))
    inc_threads = 1;
  if ((argc > 3 && strcmp("--test-small", argv[3]) == 0) ||
      (argc > 4 && strcmp("--test-small", argv[4]) == 0) ||
      (argc > 5 && strcmp("--test-small", argv[5]) == 0))
    test_small = 1;
  // Check nested parallelism availability
  omp_set_nested(1);
  if (omp_get_nested() != 1) {
    puts("Warning: Nested parallelism desired but unavailable");
  }
  // Check processors and threads
  int processors = omp_get_num_procs(); // Available processors
  printf("Array size = %lld\nProcesses = %d\nProcessors = %d\n", size, threads,
         processors);
  C *= processors;
  if (threads > processors) {
    printf("Warning: %d threads requested, will run_omp on %d processors "
           "available\n",
           threads, processors);
  }
  int max_threads = omp_get_max_threads(); // Max available threads
  if (threads > max_threads) // Requested threads are more than max available
  {
    printf("Error: Cannot use %d threads, only %d threads available\n", threads,
           max_threads);
    return 1;
  }
  // Array allocation
  int *a = malloc(sizeof(int) * size * 2);
  int *temp = malloc(sizeof(int) * size);
  if (a == NULL || temp == NULL) {
    printf("Error: Could not allocate array of size %lld\n", size);
    return 1;
  }
  printf("Will run each algorithm %d times, to calculate avg cost time\n",
         TIME_ROUND);
  // Random array initialization
  long long i;
  srand(time(NULL));
  double start = get_time();
  for (i = 0; i < size; i++) {
    a[i] = rand() % ELEMENT_MAX_RANGE;
  }
  double end = get_time();
  printf("gen data cost %.6f\n", end - start);

  /*//test only*/
  /*int test_arr[] = {12,15,6,3,2,14,11,14,4,4,4,4,5,3,7,0};*/
  /*memcpy(a, test_arr, sizeof(int) * size);*/

  int *tmp_refresh = malloc(sizeof(int) * size);
  /*memcpy(tmp_refresh, a, sizeof(int) * size);*/
  for (long long i = 0; i < size; ++i) {
    tmp_refresh[i] = a[i];
  }
  // Sort
  /*double start = get_time();*/
  if (test_small) {
    printf("With plain_parallel_merge\n");
    run_test(a, size, temp, threads, 0, tmp_refresh, run_omp, inc_threads, 1);
  } else if (!merge_type) {
    printf("With segmented parallel_merge_path\n");
    run_test(a, size, temp, threads, 1, tmp_refresh, run_segmented_merge_path,
             inc_threads, 0);
#pragma omp parallel for num_threads(threads) schedule(dynamic, size / threads)
    for (long long int i = 0; i < size; ++i) {
      a[i] = tmp_refresh[i];
      temp[i] = 0;
    }
    printf("With parallel_merge_path\n");
    run_test(a, size, temp, threads, 1, tmp_refresh, run_merge_path,
             inc_threads, 0);
#pragma omp parallel for num_threads(threads) schedule(dynamic, size / threads)
    for (long long int i = 0; i < size; ++i) {
      a[i] = tmp_refresh[i];
      temp[i] = 0;
    }
    printf("With parallel_binary_merge\n");
    run_test(a, size, temp, threads, 1, tmp_refresh, run_omp, inc_threads, 0);
#pragma omp parallel for num_threads(threads) schedule(dynamic, size / threads)
    for (long long int i = 0; i < size; ++i) {
      a[i] = tmp_refresh[i];
      temp[i] = 0;
    }
    printf("Without parallel_binary_merge\n");
    run_test(a, size, temp, threads, 0, tmp_refresh, run_omp, inc_threads, 0);
  } else if (merge_type == 1) {
    printf("Without parallel_binary_merge\n");
    run_test(a, size, temp, threads, 0, tmp_refresh, run_omp, inc_threads, 0);
  } else if (merge_type == 2) {
    printf("With parallel_binary_merge\n");
    run_test(a, size, temp, threads, 1, tmp_refresh, run_omp, inc_threads, 0);
  } else if (merge_type == 3) {
    printf("With segmented parallel_merge_path\n");
    run_test(a, size, temp, threads, 1, tmp_refresh, run_segmented_merge_path,
             inc_threads, 0);
  } else if (merge_type == 4) {
    printf("With parallel_merge_path\n");
    run_test(a, size, temp, threads, 1, tmp_refresh, run_merge_path,
             inc_threads, 0);
  }
  free(a);
  free(temp);
  return 0;
}
void run_test(int a[], long long int size, int temp[], int threads,
              int parallel_merge, int tmp_refresh[],
              void (*run_omp)(int a[], long long int size, int temp[],
                              int threads, int parallel_merge),
              int iter_threads, int test_small) {
  for (SMALL = (test_small == 1 ? 1 : SMALL); SMALL <= 4096; SMALL <<= 1) {
    printf("\nSMALL = %d\n", SMALL);
    for (int th = threads; th > 0;) {
      /*int th = threads;*/
      omp_set_num_threads(th);
      long long int i = 0;
      double start, end;
      double total_elapsed = 0;
      for (long long int i = 0; i < TIME_ROUND; ++i) {
        start = get_time();
        run_omp(a, size, temp, th, parallel_merge);
        /*double end = get_time();*/
        end = get_time();
        if (i != TIME_ROUND - 1) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, size / threads)
          for (long long int i = 0; i < size; ++i) {
            a[i] = tmp_refresh[i];
            temp[i] = 0;
          }
        }
        total_elapsed += end - start;
      }
      printf("threads: %d\t"
             "Avg Elapsed = %.6f\n",
             th, total_elapsed / TIME_ROUND);
      // Result check
      for (i = 1; i < size; i++) {
        if (!(a[i - 1] <= a[i])) {
          printf("Implementation error: a[%lld]=%d > a[%lld]=%d\n", i - 1,
                 a[i - 1], i, a[i]);
          exit(1);
        }
      }
      puts("-Success-\n");
#pragma omp parallel for num_threads(threads) schedule(dynamic, size / threads)
      for (long long int i = 0; i < size; ++i) {
        a[i] = tmp_refresh[i];
        temp[i] = 0;
      }
      if (!iter_threads)
        break;
      if (th == 80) {
        th = 64;
      } else if (th == 128) {
        th = 80;
      } else {
        th >>= 1;
      }
    }
    if (!test_small)
      break;
  }
}

// Driver
void run_omp(int a[], long long int size, int temp[], int threads,
             int parallel_merge) {
  // Enable nested parallelism, if available
  omp_set_nested(1);
  // Parallel mergesort
  mergesort_parallel_omp(a, size, temp, threads, parallel_merge);
}

// OpenMP merge sort with given number of threads
void mergesort_parallel_omp(int a[], long long int size, int temp[],
                            int threads, int parallel_merge) {
  if (threads == 1) {
    mergesort_serial(a, size, temp);
  } else if (threads > 1) {
#pragma omp parallel sections num_threads(threads)
    {
#pragma omp section
      {
        mergesort_parallel_omp(a, size / 2, temp, threads / 2, parallel_merge);
      }
#pragma omp section
      {
        mergesort_parallel_omp(a + size / 2, size - size / 2, temp + size / 2,
                               threads - threads / 2, parallel_merge);
      }
      /*
       *#pragma omp section
       *            merge(a, size, temp);
       */
    }
    // Thread allocation is implementation dependent
    // Some threads can execute multiple sections while others are idle
    // Merge the two sorted sub-arrays through temp
    if (parallel_merge != 0 && pow(2, threads) >= size)
      merge_binary(a, size, temp, threads);
    else
      merge_serial(a, size, temp);
  } else {
    printf("Error: %d threads\n", threads);
    return;
  }
}

void mergesort_serial(int a[], long long int size, int temp[]) {
  // Switch to insertion sort for small arrays
  if (size <= SMALL) {
    insertion_sort(a, size);
    return;
  }
  mergesort_serial(a, size / 2, temp);
  mergesort_serial(a + size / 2, size - size / 2, temp);
  // Merge the two sorted subarrays into a temp array
  merge_serial(a, size, temp);
}

void merge_binary(int a[], long long int size, int temp[], int threads) {
  long long int i1 = 0;
  long long int i2 = 0;
  long long int tempi = 0;
#pragma omp parallel for num_threads(threads)
  for (i1 = 0; i1 < size / 2; ++i1) {
    tempi = binsearch_findfirst_ge(a + size / 2, size - size / 2, a[i1]);
    temp[i1 + tempi] = a[i1];
  }

#pragma omp parallel for num_threads(threads)
  for (i2 = 0; i2 < size - size / 2; ++i2) {
    tempi = binsearch_findfirst_g(a, size / 2, a[i2 + size / 2]);
    temp[i2 + tempi] = a[i2 + size / 2];
  }

#pragma omp parallel for num_threads(threads)
  for (int i = 0; i < size; ++i)
    a[i] = temp[i];
}

int binsearch_findfirst_ge(int *array, long long int n, int target) {
  // first greater or equal than target
  long long int start = 0, end = n - 1;
  while (start <= end) {
    long long int mid = (start + end) / 2;
    if (array[mid] >= target)
      end = mid - 1;
    else
      start = mid + 1;
  }
  return start;
}

int binsearch_findfirst_g(int *array, long long int n, int target) {
  // first greater
  long long int start = 0, end = n - 1;
  while (start <= end) {
    long long int mid = (start + end) / 2;
    if (array[mid] > target)
      end = mid - 1;
    else
      start = mid + 1;
  }
  return start;
}
void insertion_sort(int a[], long long int size) {
  long long int i;
  for (i = 0; i < size; i++) {
    int j, v = a[i];
    for (j = i - 1; j >= 0; j--) {
      if (a[j] <= v)
        break;
      a[j + 1] = a[j];
    }
    a[j + 1] = v;
  }
}
void merge_serial(int a[], long long int size, int temp[]) {
  long long int i1 = 0;
  long long int i2 = size / 2;
  int tempi = 0;
  while (i1 < size / 2 && i2 < size) {
    if (a[i1] < a[i2]) {
      temp[tempi] = a[i1];
      i1++;
    } else {
      temp[tempi] = a[i2];
      i2++;
    }
    tempi++;
  }
  while (i1 < size / 2) {
    temp[tempi] = a[i1];
    i1++;
    tempi++;
  }
  while (i2 < size) {
    temp[tempi] = a[i2];
    i2++;
    tempi++;
  }
  // Copy sorted temp array into main array, a
  memcpy(a, temp, size * sizeof(int));
}
void merge_serial_general(int a[], int b[], long long int astart,
                          long long int bstart, int temp[],
                          long long int tempstart, long long int size,
                          long long int asize, long long int bsize,
                          long long int *ainc, long long int *binc) {
  long long int i1 = astart;
  long long int i2 = bstart;
  int tempi = 0;
  while (tempi < size && i1 < asize && i2 < bsize) {
    if (a[i1] < b[i2]) {
      temp[tempi + tempstart] = a[i1];
      i1++;
    } else {
      temp[tempi + tempstart] = b[i2];
      i2++;
    }
    tempi++;
  }
  while (tempi < size && i1 < asize) {
    temp[tempi + tempstart] = a[i1];
    i1++;
    tempi++;
  }
  while (tempi < size && i2 < bsize) {
    temp[tempi + tempstart] = b[i2];
    i2++;
    tempi++;
  }
  if (ainc != NULL)
    *ainc = i1;
  if (binc != NULL)
    *binc = i2;
}

// merge path
// Driver
void run_segmented_merge_path(int a[], long long int size, int temp[],
                              int threads, int parallel_merge) {
  // Enable nested parallelism, if available
  omp_set_nested(1);
  // Parallel mergesort
  segmented_mergesort_merge_path(a, size, temp, threads);
}
void segmented_mergesort_merge_path(int a[], long long int size, int temp[],
                                    int threads) {
  if (threads == 1 || size <= SMALL) {
    mergesort_serial(a, size, temp);
  } else if (threads > 1) {
#pragma omp parallel sections num_threads(threads)
    {
#pragma omp section
      { mergesort_merge_path(a, size / 2, temp, threads / 2); }
#pragma omp section
      {
        mergesort_merge_path(a + size / 2, size - size / 2, temp + size / 2,
                             threads - threads / 2);
      }
    }
    // Thread allocation is implementation dependent
    // Some threads can execute multiple sections while others are idle
    // Merge the two sorted sub-arrays through temp
    segmented_merge_path_parallel(a, size, temp, threads);
  } else {
    printf("Error: %d threads\n", threads);
    return;
  }
}
void run_merge_path(int a[], long long int size, int temp[], int threads,
                    int parallel_merge) {
  // Enable nested parallelism, if available
  omp_set_nested(1);
  // Parallel mergesort
  mergesort_merge_path(a, size, temp, threads);
}
void mergesort_merge_path(int a[], long long int size, int temp[],
                          int threads) {
  if (threads == 1 || size <= SMALL) {
    mergesort_serial(a, size, temp);
  } else if (threads > 1) {
#pragma omp parallel sections num_threads(threads)
    {
#pragma omp section
      { mergesort_merge_path(a, size / 2, temp, threads / 2); }
#pragma omp section
      {
        mergesort_merge_path(a + size / 2, size - size / 2, temp + size / 2,
                             threads - threads / 2);
      }
    }
    // Thread allocation is implementation dependent
    // Some threads can execute multiple sections while others are idle
    // Merge the two sorted sub-arrays through temp
    merge_path_parallel(a, size, temp, threads);
  } else {
    printf("Error: %d threads\n", threads);
    return;
  }
}
void segmented_merge_path_parallel(int a[], long long int size, int temp[],
                                   int threads) {
  long long int l = size / 2;
  long long int l = C / 2;
  int maxiter = 2;
  maxiter = (maxiter == 0 ? 1 : maxiter);
  long long int startPoint = 0;
  long long int a_offset = 0;
  long long int b_offset = size / 2;
  int i = 0;
  int p = threads;
  long long int ainc, binc;
  for (int k = 0; k < maxiter; ++k) {
    if (k == maxiter - 1) {
      l = size - k * l;
    }
#pragma omp parallel for num_threads(threads) schedule(static, 1)
    for (i = 0; i < threads; ++i) {
      long long int length =
          (i == threads - 1 ? (l - ((l / p) * (p - 1))) : l / p);
      long long int astart;
      long long int bstart;
      long long int diag = i * (l / p);
      diagnal_intersection(
          a + a_offset, a + b_offset,
          ((l > size / 2 - a_offset) ? (size / 2 - a_offset) : l),
          ((l > (size - b_offset)) ? ((size - b_offset)) : l), threads, &astart,
          &bstart, i, diag);
      long long int tempstart = startPoint + (i) * (l / p);
      merge_serial_general(a + a_offset, a + b_offset, astart, bstart, temp,
                           tempstart, length, size / 2 - a_offset,
                           size - b_offset, (i == threads - 1) ? &ainc : NULL,
                           (i == threads - 1) ? &binc : NULL);
    }
    a_offset += ainc;
    b_offset += binc;
    startPoint += l;
  }
#pragma omp parallel for num_threads(threads) schedule(dynamic, size / threads)
  for (long long int j = 0; j < size; ++j) {
    a[j] = temp[j];
  }
}
void merge_path_parallel(int a[], long long int size, int temp[], int threads) {
  long long int i = 0;
  int p = threads;
#pragma omp parallel for num_threads(threads) schedule(static, 1)
  for (i = 0; i < threads; ++i) {
    long long int length =
        (i == threads - 1 ? (size - ((size / p) * (p - 1))) : size / p);
    long long int astart;
    long long int bstart;
    long long int ainc, binc;
    long long int diag = i * (size / p);
    diagnal_intersection(a, a + size / 2, size / 2, size - size / 2, threads,
                         &astart, &bstart, i, diag);
    long long int tempstart = (i) * (size / p);
    merge_serial_general(a, a + size / 2, astart, bstart, temp, tempstart,
                         length, size / 2, size - size / 2, &ainc, &binc);
  }
#pragma omp parallel for num_threads(threads) schedule(dynamic, size / threads)
  for (long long int j = 0; j < size; ++j) {
    a[j] = temp[j];
  }
}
void diagnal_intersection(int a[], int b[], long long int asize,
                          long long int bsize, int threads,
                          long long int *astart, long long int *bstart,
                          long long int i, long long int diag) {
  if (i == 0) {
    *astart = 0;
    *bstart = 0;
    return;
  }
  int overhalf_a = diag > asize ? 1 : 0;
  int overhalf_b = diag > bsize ? 1 : 0;
  long long int overhalf_offset_b = ((overhalf_a == 0) ? 0 : (diag - asize));
  long long int overhalf_offset_a = ((overhalf_b == 0) ? 0 : (diag - bsize));
  if (overhalf_a && overhalf_b) {
    diag = bsize + asize - diag;
  } else if (overhalf_a && !overhalf_b) {
    diag = asize;
  } else if (overhalf_b && !overhalf_a) {
    diag = bsize;
  }
  long long int left = 0;
  long long int right = diag;
  long long int ai = -1, bi = -1;
  while (left <= right) {
    int mid = (left + right) / 2;
    ai = diag - mid + overhalf_offset_a;
    bi = mid + overhalf_offset_b;
    *astart = ai;
    *bstart = bi;
    if ((bi >= bsize && a[ai] > b[bi - 1]) ||
        (ai >= asize && a[ai - 1] < b[bi]))
      break;
    if (a[ai] >= b[bi - 1] && a[ai - 1] < b[bi])
      break;
    else if (ai == asize || a[ai] >= b[bi - 1])
      left = mid + 1;
    else
      right = mid - 1;
  }
  return;
}
