#!/bin/bash
# 增量编译并运行 Inference Service（保留 build 缓存）
cd "$(dirname "$0")"
mkdir -p build && cd build
cmake .. && make
cd ..
./build/inference_service