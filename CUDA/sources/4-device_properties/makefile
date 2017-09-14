.PHONY: all clean
NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3 -I../include
LINK=-lcublas -lcudart

CUDA_EXE=task5 task5_solution

all: $(CUDA_EXE)

task5: task5.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

task5_solution: task5_solution.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

clean:
	rm -f $(CUDA_EXE)
