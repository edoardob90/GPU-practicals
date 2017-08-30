.PHONY: all clean

vpath %.hpp include
VPATH = $(wildcard task*)

NVCC_INC=-I ./include
NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3 -g -G
NVCC_LINK=-lcublas -lcuda

CUDA_BENCHMARKS=task1 \
				task2 \
				task3 \
				task4 \
				task5 \
				task6 \
				task7

all : $(CUDA_BENCHMARKS)

task1 : task1.cu
	nvcc $(NVCC_INC) $< -o task1.x $(NVCC_INC)

clean : 
	rm -f $(CUDA_BENCHMARKS) *.nvvp
