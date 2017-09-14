.PHONY: all clean
NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3 -I../include
LINK=-lcublas -lcudart

CUDA_EXE=task2 task2_solution

all: $(CUDA_EXE)

task2: task2.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

task2_solution: task2_solution.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

clean:
	rm -f $(CUDA_EXE)
