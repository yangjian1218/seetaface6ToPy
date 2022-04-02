#/bin/env bash

rm -rf seetaface/build && mkdir -p seetaface/build && cd seetaface/build && cmake .. && make && cd ../../

export DYLD_FALLBACK_LIBRARY_PATH=`pwd`/seetaface/lib_mac:$DYLD_FALLBACK_LIBRARY_PATH

python3 seeta_test.py

