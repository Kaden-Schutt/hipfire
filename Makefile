.PHONY: build install

build:
	cargo build --release

install:
	./install.sh
