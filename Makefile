TARGET_EXEC ?= rbm

BUILD_DIR ?= build
SRC_DIRS ?= src
INC_DIRS ?= include

# PFAPACK
INC_DIRS += pfapack/c_interface
PFAPACK = $(BUILD_DIR)/pfapack/libpfapack.a
PFAPACKC = $(BUILD_DIR)/pfapack/libcpfapack.a

DESTDIR ?= /usr/local/bin/

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -type f)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS += /usr/local/include/eigen3/
INC_DIRS += /usr/local/include
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS += $(INC_FLAGS) -MMD -MP -Wall -std=c++14 -O3 -march=native

OMP = -fopenmp

ifeq ($(notdir $(CXX)), clang++)
LDFLAGS += -L/usr/local/opt/llvm/lib/
endif

ifneq (,$(findstring gcc,$(CXX)))
LDFLAGS := -L$(HOME)/boost-gcc/lib $(LDFLAGS)
CPPFLAGS := -I$(HOME)/boost-gcc/include $(CPPFLAGS)
endif

LDFLAGS += -lboost_program_options

LDFLAGS += -lblas -llapack $(PFAPACK) $(PFAPACKC)

install: all
	cp $(BUILD_DIR)/$(TARGET_EXEC) $(DESTDIR)/$(TARGET_EXEC)
	cp plot-on-the-go/potg $(DESTDIR)/potg

all: $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS) $(PFAPACK) $(PFAPACKC)
	@echo "[ LD ] $@ $(LDFLAGS)"
	@$(CXX) $(OMP) $(LDFLAGS) $(OBJS) -o $@ 
# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	@$(MKDIR_P) $(dir $@)
	@echo "[ $(notdir $(CXX)) ] $<"
	@$(CXX) $(OMP) $(CPPFLAGS) -c $< -o $@

.PHONY: install all clean

clean:
	$(RM) -r $(BUILD_DIR)

$(PFAPACK):
	cd pfapack/fortran/ && make
	@mkdir -p $(BUILD_DIR)/pfapack
	cp pfapack/fortran/libpfapack.a $(PFAPACK)

$(PFAPACKC):
	cd pfapack/c_interface && make
	@mkdir -p $(BUILD_DIR)/pfapack
	cp pfapack/c_interface/libcpfapack.a $(PFAPACKC)

build_pfapack: $(PFAPACK) $(PFAPACKC)

clean_pfapack:
	cd pfapack/fortran/ && make clean
	cd pfapack/c_interface && make clean
	rm -r build/pfapack

doc:
	mv README.md README.md.tmp
	./.md2dox
	cd docs && make html && cd ..
	mv README.md.tmp README.md

lsp:
	@compiledb make --always-make --dry-run


-include $(DEPS)

MKDIR_P ?= mkdir -p

