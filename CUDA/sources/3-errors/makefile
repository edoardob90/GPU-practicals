.PHONY: all clean
NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3 -I../include
LINK=-lcublas -lcudart

CUDA_EXE=task4 task4_solution

all: $(CUDA_EXE)

task4: task4.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

task4_solution: task4_solution.cu
	nvcc $(NVCC_FLAGS) $^ -o $@ $(LINK) 

clean:
	rm -f $(CUDA_EXE)
