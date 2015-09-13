CFLAGS += -std=c99 -pedantic

CPPFLAGS += -Wall -Wextra -Werror

LDLIBS +=-lOpenCL

reduction_test: reduction_test.c
