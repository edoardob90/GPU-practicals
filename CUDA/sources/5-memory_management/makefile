.PHONY: all clean
NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3 -I../include
LINK=-lcublas -lcudart

CUDA_EXE=task6 task6_solution

all: $(CUDA_EXE)

task6: task6.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

task6_solution: task6_solution.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

clean:
	rm -f $(CUDA_EXE)
