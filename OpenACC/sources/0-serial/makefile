.PHONY: all clean

VERBOSE ?= 0
override CPPFLAGS += -I../common

CXX = pgcc
FC  = pgf90
CXXFLAGS = -DPGI -O3 -acc -Minfo=accel -ta=tesla:cc35
FCFLAGS  = -O3 -acc -Minfo=accel -ta=tesla:cc35
LDFLAGS += -L$(CUDA_LIBRARY) -lcudart -ta=tesla:cc35

OPENACC_BENCHMARKS=task1

BENCHMARKS=$(OPENACC_BENCHMARKS)

all: $(BENCHMARKS)

task1: task1.c
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod
