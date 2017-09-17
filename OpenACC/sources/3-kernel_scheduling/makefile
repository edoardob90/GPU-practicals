.PHONY: all clean openmp

VERBOSE ?= 0
override CPPFLAGS += -I../common

CXX = pgcc
FC  = pgf90
CXXFLAGS = -DPGI -O3 -acc -Minfo=accel -ta=tesla:cc35
FCFLAGS  = -O3 -acc -Minfo=accel -ta=tesla:cc35
LDFLAGS += -L$(CUDA_LIBRARY) -lcudart -ta=tesla:cc35

OPENACC_BENCHMARKS=task4.x task4_solution.x task4_4096_solution.x task4_4096_v3.x

all: $(OPENACC_BENCHMARKS)

openmp: task4_4096_omp.x

task4_4096_omp.x: task4_4096_omp.c
	$(CXX) -DPGI -O3 -mp -Minfo=all $< -o $@

%.x: %.c
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OPENACC_BENCHMARKS) *.cub *.ptx *.o *.mod
