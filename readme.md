# Fast Parallel Merge Sort With Merge Path

OpenMP parallelized merge sort optimized with merge sort algorithm (for the record, this seems to be the only version available on github that implemented merge sort algorithm with OpenMP). Traditional parallel merge sort cannot efficiently parallel merge process. With merge path algorithm, we can achieve O(n/d) time complexity (traditional parallel merge sort can only achieve O(n) during the last few merge steps) on merge process. More detail about merge sort algorithm can be found: https://arxiv.org/abs/1406.2628 . This code also implemented plain, binary, segmented merge path and unsegmented merge path versions of parallel merge sort.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have gcc and make in your PATH.

### Installing

Clone this repository.

```
git clone https://github.com/GlenGGG/FastParallelMergeSort.git
cd FastParallelMergeSort
```

Build the project

```
make
```

### Usage

Simply run the code.

```
./omp_mergesort [num_threads] [array_length] [--merge-type-{plain, binary, mergepath, segmented-mergepath}] [--inc-threads]
```

For example, run this code to get an insight:

```
./omp_mergesort 16 20000000 --inc-threads
```

