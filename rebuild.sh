#!/bin/bash
# 增量编译并运行 Inference Service（保留 build 缓存）
cd "$(dirname "$0")"

# 解析命令行选项
BUILD_TESTS=ON
for arg in "$@"; do
    case $arg in
        --no-test) BUILD_TESTS=OFF ;;
        --test)    BUILD_TESTS=ON ;;
    esac
done

mkdir -p build && cd build
cmake .. -DBUILD_TESTS=$BUILD_TESTS && make
cd ..
./build/inference_service