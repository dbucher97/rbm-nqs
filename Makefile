TARGET_EXEC ?= rbm
TARGET_UNITTEST ?= unittest

BUILD_DIR ?= build
SRC_DIRS ?= src
TEST_DIR ?= test
INC_DIRS ?= include

MAT_BACKEND ?= EIGEN

# PFAPACK
INC_DIRS += lib/pfapack/c_interface
PFAPACK = $(BUILD_DIR)/pfapack/libpfapack.a
PFAPACKC = $(BUILD_DIR)/pfapack/libcpfapack.a

# MINRES QLP
INC_DIRS += lib/minresqlp
INC_DIRS += $(HOME)/googletest/googletest/include/
MINRESQLP = $(BUILD_DIR)/minresqlp/libminresqlp.a

DESTDIR ?= $(HOME)/.local/bin/

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -type f)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

TEST_SRCS := $(shell find $(TEST_DIR) -name *.cpp -type f)
TEST_OBJS := $(TEST_SRCS:%=$(BUILD_DIR)/%.o)
TEST_OBJS += $(filter-out $(BUILD_DIR)/src/main.cpp.o,$(OBJS))

# INDCLUDE DIRS
INC_DIRS += $(HOME)/eigen3/
INC_DIRS += $(HOME)/OpenBLAS/
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS += $(INC_FLAGS) $(MKL_INC) -MMD -MP -Wall -std=c++14 -O3 -march=native
CPPFLAGS += -DITERATIVE_USE_MKL

OMP = -fopenmp

# HANDLE DIFFERENT MATH BACKENDS
ifeq ($(MAT_BACKEND),BLAS)
CPPFLAGS += -DEIGEN_USE_BLAS
$(info using BLAS)
else ifeq ($(MAT_BACKEN),MKL)
CPPFLAGS += -DEIGEN_USE_MKL_ALL
$(info using MKL)
else
$(info using EIGEN)
endif

LDFLAGS += -lboost_program_options

LDFLAGS += -lgfortran

LDFLAGS += $(MKL_LIB)

TEST_LDFLAGS = libgtest.a libgtest_main.a
TEST_LDFLAGS := $(addprefix $(HOME)/googletest/build/lib/,$(TEST_LDFLAGS))

install: all
	cp $(BUILD_DIR)/$(TARGET_EXEC) $(DESTDIR)/$(TARGET_EXEC)

all: $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS) $(PFAPACK) $(PFAPACKC) $(MINRESQLP)
	@echo "[ LD ] $@ $(LDFLAGS)"
	$(CXX) $(OMP) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/$(TARGET_UNITTEST): $(TEST_OBJS) $(PFAPACK) $(PFAPACKC) $(MINRESQLP)
	@echo "[ LD ] $@"
	@$(CXX) $(OMP) $(LDFLAGS) $(TEST_LDFLAGS) $^ -o $@ 
# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	@$(MKDIR_P) $(dir $@)
	@echo "[ $(notdir $(CXX)) ] $<"
	@$(CXX) $(OMP) $(CPPFLAGS) -c $< -o $@

.PHONY: install all clean

test: $(BUILD_DIR)/$(TARGET_UNITTEST)
	@$(BUILD_DIR)/$(TARGET_UNITTEST)

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


-include $(DEPS)

MKDIR_P ?= mkdir -p

