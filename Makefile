TARGET_EXEC ?= rbm

BUILD_DIR ?= build
SRC_DIRS ?= src
INC_DIRS ?= include

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -type f)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS += /usr/local/include/
INC_DIRS += /usr/local/include/eigen3/
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP -Wall -std=c++14

OMP = -fopenmp
ifeq ($(notdir $(CXX)), clang++)
LDFLAGS += -L/usr/local/opt/llvm/lib/
endif


all: $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	@echo "[ LD ] $@ $(LDFLAGS)"
	@$(CXX) $(OMP) $(OBJS) -o $@ $(LDFLAGS)

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	@$(MKDIR_P) $(dir $@)
	@echo "[ $(notdir $(CXX)) ] $<"
	@$(CXX) $(OMP) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: all clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p

