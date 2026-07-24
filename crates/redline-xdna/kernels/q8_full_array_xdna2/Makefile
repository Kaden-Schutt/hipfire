PEANO_INSTALL_DIR ?= $(shell python3 -c "import importlib.metadata as m; print(m.distribution('llvm-aie').locate_file('llvm-aie'))")
MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")

BUILD_DIR ?= build
BFP16_EMULATION ?= 1
PROFILE_FLAGS ?=

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined \
	-Wno-empty-body -Wno-missing-template-arg-list-after-template-kw
AIE2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf \
	$(WARNING_FLAGS) -DNDEBUG -I $(MLIR_AIE_DIR)/include
ifeq ($(BFP16_EMULATION),1)
AIE2P_FLAGS += -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1
endif

.PHONY: all clean

all: $(BUILD_DIR)/final_q8_full_array.xclbin \
	$(BUILD_DIR)/insts_q8_full_array.bin

$(BUILD_DIR)/q8_full_array.cc.o: q8_full_array.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(AIE2P_FLAGS) \
		$(PROFILE_FLAGS) -c $(abspath $<) -o $(@F)

$(BUILD_DIR)/q8_full_array.mlir: q8_full_array_iron.py
	mkdir -p $(@D)
	python3 $< > $@

$(BUILD_DIR)/final_q8_full_array.xclbin \
$(BUILD_DIR)/insts_q8_full_array.bin &: \
		$(BUILD_DIR)/q8_full_array.mlir \
		$(BUILD_DIR)/q8_full_array.cc.o
	cd $(BUILD_DIR) && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final_q8_full_array.xclbin \
		--npu-insts-name=insts_q8_full_array.bin q8_full_array.mlir

clean:
	rm -rf $(BUILD_DIR)
