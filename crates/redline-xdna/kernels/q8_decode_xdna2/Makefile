PEANO_INSTALL_DIR ?= $(shell python3 -c "import importlib.metadata as m; print(m.distribution('llvm-aie').locate_file('llvm-aie'))")
MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")

K ?= 2048
BUILD_DIR ?= build

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined \
	-Wno-empty-body -Wno-missing-template-arg-list-after-template-kw
AIE2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf \
	$(WARNING_FLAGS) -DNDEBUG -I $(MLIR_AIE_DIR)/include

.PHONY: all clean

all: $(BUILD_DIR)/final_q8_decode.xclbin $(BUILD_DIR)/insts_q8_decode.bin

$(BUILD_DIR)/q8_decode.cc.o: q8_decode.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(AIE2P_FLAGS) \
		-c $(abspath $<) -o $(@F)

$(BUILD_DIR)/q8_decode.mlir: q8_decode_iron.py
	mkdir -p $(@D)
	python3 $< -K $(K) > $@

$(BUILD_DIR)/final_q8_decode.xclbin $(BUILD_DIR)/insts_q8_decode.bin &: \
		$(BUILD_DIR)/q8_decode.mlir $(BUILD_DIR)/q8_decode.cc.o
	cd $(BUILD_DIR) && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=final_q8_decode.xclbin \
		--npu-insts-name=insts_q8_decode.bin q8_decode.mlir

clean:
	rm -rf $(BUILD_DIR)
