LDLIBS=-lm -lOpenCL

reduction_test: reduction_test.cc
	$(LINK.cc) $< $(LOADLIBES) $(LDLIBS) -o $@
