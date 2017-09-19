DIRS := $(shell find . -type d -name "*-*")

.PHONY: clean $(DIRS)

# Global clean
clean: $(DIRS)

$(DIRS):
	$(MAKE) -C $@ clean
