CC=gcc
#OFLAGS=-O3 -g
OFLAGS= -O3
WFLAGS=-Wall -Werror
LDFLAGS=-lm
CFLAGS=$(OFLAGS) $(WFLAGS)
MPIFLAGS=
OMPFLAGS=-fopenmp
UPCFLAGS=

SRC :=	omp_mergesort.c

ALL :=  $(foreach src,$(SRC),$(subst .upc,,$(subst .c,,$(src))))

default: $(ALL)

tags: $(SRC)
	ctags $^


omp_mergesort: omp_mergesort.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	@- rm -f $(ALL) tags
