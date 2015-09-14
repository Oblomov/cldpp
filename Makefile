CFLAGS += -std=c99 -pedantic
CPPFLAGS += -Wall -Wextra -Werror
LDLIBS +=-lOpenCL
SRCDIR = src
TARGETS = $(foreach source,$(wildcard $(SRCDIR)/*.c),\
	  $(basename $(notdir $(source))))

vpath %.c $(SRCDIR)

all: $(TARGETS)

$(TARGETS):

clean:
	@rm -rf $(TARGETS)
