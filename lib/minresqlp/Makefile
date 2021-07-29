FC    ?=  gfortran
# FFLAGS = -g -O0 -pedantic -Wall -W -fbounds-check
FFLAGS = -g -O3

BUILD_DIR = build
SRC_DIR = Src

OBJS = minresqlpDataModule.o zminresqlpDataModule.o zminresqlpBlasModule.o minresqlpBlasModule.o zminresqlpModule.o minresqlpModule.o
OBJS := $(addprefix $(BUILD_DIR)/,$(OBJS))

ifeq ($(FC),ifort)
FFLAGS += -module $(BUILD_DIR) -I$(BUILD_DIR)
else
FFLAGS += -J$(BUILD_DIR) -I$(BUILD_DIR)
endif

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.f90
	@mkdir -p $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/libminresqlp.a: $(OBJS)
	$(AR) ru $@ $(OBJS)

clean:
	rm -rf build
