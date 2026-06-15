#!/bin/bash
# 全量重建并运行 Inference Service
cd "$(dirname "$0")"

# 解析命令行选项
BUILD_TESTS=ON
for arg in "$@"; do
    case $arg in
        --no-test) BUILD_TESTS=OFF ;;
        --test)    BUILD_TESTS=ON ;;
    esac
done

rm -rf build
mkdir build && cd build
cmake .. -DBUILD_TESTS=$BUILD_TESTS && make
cd ..
./build/inference_service