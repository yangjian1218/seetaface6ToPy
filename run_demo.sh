#/bin/env bash

# build *.dylib for mac_x64
rm -rf SeetaFace6OpenIndex/ && \
git clone --recursive git@github.com:frkhit/SeetaFace6OpenIndex.git SeetaFace6OpenIndex && \
cd  SeetaFace6OpenIndex && \
bash build_mac.sh && cd .. && \
mkdir -p seetaface/lib_mac/ && cp SeetaFace6OpenIndex/build/lib64/* seetaface/lib_mac/


# build SeetaFaceAPI
rm -rf seetaface/build && mkdir -p seetaface/build && cd seetaface/build && cmake .. && make && cd ../../

export DYLD_FALLBACK_LIBRARY_PATH=`pwd`/seetaface/lib_mac:$DYLD_FALLBACK_LIBRARY_PATH

python3 seeta_test.py

