DIRS=0-serial 3-kernel_scheduling 2-data_movement 1-kernels

.PHONY: clean $(DIRS)

# Global clean
clean: $(DIRS)

$(DIRS):
	$(MAKE) -C $@ clean
