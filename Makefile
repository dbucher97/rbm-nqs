TARGET_EXEC ?= rbm

BUILD_DIR ?= build
SRC_DIRS ?= src
INC_DIRS ?= include

MAT_BACKEND ?= EIGEN

# PFAPACK
INC_DIRS += lib/pfapack/c_interface
PFAPACK = $(BUILD_DIR)/pfapack/libpfapack.a
PFAPACKC = $(BUILD_DIR)/pfapack/libcpfapack.a

# MINRES QLP
INC_DIRS += lib/minresqlp
MINRESQLP = $(BUILD_DIR)/minresqlp/libminresqlp.a

DESTDIR ?= /usr/local/bin/

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -type f)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

# INDCLUDE DIRS
INC_DIRS += /usr/local/include/eigen3/
INC_DIRS += /usr/local/opt/openblas/include/
INC_DIRS += /usr/local/include/
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

# HANDLE DIFFERENT MATH BACKENDS
ifeq ($(MAT_BACKEND),BLAS)
CPPFLAGS += -DEIGEN_USE_BLAS
$(info using BLAS)
else ifeq ($(MAT_BACKEN),MKL)
ifeq ($(MKLROOT),)
$(error MKL ENV not sourced! Run source /opt/intel/oneapi/mkl/latest/env/vars.sh)
endif
CPPFLAGS += -DEIGEN_USE_MKL_ALL -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
LDFLAGS += -L/opt/intel/oneapi/compiler/latest/mac/compiler/lib/
LDFLAGS += -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl 
$(info using MKL)
else
$(info using EIGEN)
endif

# LINK BLAS IF NOT USE MKL
ifneq ($(METHOD),MKL)
LDFLAGS += -L/usr/local/opt/openblas/lib/
LDFLAGS += -lblas -llapack -lopenblas -lpthread
endif

LDFLAGS += -lboost_program_options

LDFLAGS += $(PFAPACK) $(PFAPACKC) $(MINRESQLP)

install: all
	cp $(BUILD_DIR)/$(TARGET_EXEC) $(DESTDIR)/$(TARGET_EXEC)
	cp plot-on-the-go/potg $(DESTDIR)/potg

all: $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS) $(PFAPACK) $(PFAPACKC) $(MINRESQLP)
	@echo "[ LD ] $@"
	@$(CXX) $(OMP) $(LDFLAGS) $^ -o $@ 
# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	@$(MKDIR_P) $(dir $@)
	@echo "[ $(notdir $(CXX)) ] $<"
	@$(CXX) $(OMP) $(CPPFLAGS) -c $< -o $@

.PHONY: install all clean

clean:
	$(RM) -r $(BUILD_DIR)

$(PFAPACK):
	cd lib/pfapack/fortran/ && make
	@mkdir -p $(BUILD_DIR)/pfapack
	cp lib/pfapack/fortran/libpfapack.a $(PFAPACK)

$(PFAPACKC):
	cd lib/pfapack/c_interface && make
	@mkdir -p $(BUILD_DIR)/pfapack
	cp lib/pfapack/c_interface/libcpfapack.a $(PFAPACKC)

$(MINRESQLP):
	cd lib/minresqlp/ && make
	@mkdir -p $(BUILD_DIR)/minresqlp
	cp lib/minresqlp/build/$(notdir $(MINRESQLP)) $(MINRESQLP)

build_pfapack: $(PFAPACK) $(PFAPACKC)

build_minresqlp: $(MINRESQLP)

clean_pfapack:
	cd lib/pfapack/fortran/ && make clean
	cd lib/pfapack/c_interface && make clean
	rm -r build/pfapack

clean_minresqlp:
	cd lib/minresqlp/ && make clean
	rm -r build/minresqlp

doc:
	mv README.md README.md.tmp
	./.md2dox
	cd docs && make html && cd ..
	mv README.md.tmp README.md

lsp:
	@compiledb make --always-make --dry-run


-include $(DEPS)

MKDIR_P ?= mkdir -p

