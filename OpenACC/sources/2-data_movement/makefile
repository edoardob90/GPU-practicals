.PHONY: all clean

VERBOSE ?= 0
override CPPFLAGS += -I../common

CXX = pgcc
FC  = pgf90
CXXFLAGS = -DPGI -O3 -acc -Minfo=accel -ta=tesla:cc35
FCFLAGS  = -O3 -acc -Minfo=accel -ta=tesla:cc35
LDFLAGS += -L$(CUDA_LIBRARY) -lcudart -ta=tesla:cc35

OPENACC_BENCHMARKS=task3.x task3_solution.x task3_solution_schedule.x

BENCHMARKS=$(OPENACC_BENCHMARKS)

all: $(BENCHMARKS)

%.x: %.c
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod
