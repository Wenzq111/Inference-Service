#!/bin/bash
# 全量重建并运行 Inference Service
cd "$(dirname "$0")"
rm -rf build
mkdir build && cd build
cmake .. && make
cd ..
./build/inference_service