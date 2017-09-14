.PHONY: all clean
NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3 -I../include
LINK=-lcublas -lcudart

CUDA_EXE=task3 task3_solution

all: $(CUDA_EXE)

task3: task3.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

task3_solution: task3_solution.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

clean:
	rm -f $(CUDA_EXE)
