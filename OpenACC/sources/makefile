.PHONY: all clean

DIRS=0-serial 3-kernel_scheduling 2-data_movement 1-kernels

# Global clean
clean:
	@$(foreach dir,$(DIRS),echo "Cleaning in dir $(dir)"; make -C $(dir) clean)
