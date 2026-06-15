cmake_minimum_required(VERSION 3.15)
project(Inference-Service VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 下载 cpp-httplib ----
include(FetchContent)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.0
)
FetchContent_MakeAvailable(httplib)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ---- 查找 onnxruntime：优先 CMake config，失败则手动搜索 ----
find_package(onnxruntime QUIET)
if(onnxruntime_FOUND)
    set(ORT_LINK_TARGET onnxruntime::onnxruntime)
    message(STATUS "onnxruntime found via CMake config")
else()
    # 支持用户通过 -DONNXRUNTIME_ROOT=... 或环境变量指定安装根目录
    if(NOT ONNXRUNTIME_ROOT)
        set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
    endif()

    # 搜索路径：用户指定 > Homebrew > 系统标准路径
    set(_ORT_SEARCH_PATHS
        ${ONNXRUNTIME_ROOT}
        ${ONNXRUNTIME_ROOT}/include
        /opt/homebrew
        /usr/local
        /usr
    )

    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS ${_ORT_SEARCH_PATHS}
        PATH_SUFFIXES include/onnxruntime include)

    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime onnxruntime.lib
        PATHS ${_ORT_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64)

    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        set(ORT_LINK_TARGET ${ONNXRUNTIME_LIB})
        message(STATUS "onnxruntime found: include=${ONNXRUNTIME_INCLUDE_DIR}, lib=${ONNXRUNTIME_LIB}")
    else()
        message(FATAL_ERROR
            "onnxruntime not found.\n"
            "Options:\n"
            "  1. Install via package manager (brew install onnxruntime / vcpkg install onnxruntime)\n"
            "  2. Specify path: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..\n"
            "  3. Set environment variable: export ONNXRUNTIME_ROOT=/path/to/onnxruntime")
    endif()
endif()

# ---- 查找 ncnn：优先 CMake config，失败则手动搜索 ----
find_package(ncnn QUIET)
if(ncnn_FOUND)
    set(NCNN_LINK_TARGET ncnn)
    message(STATUS "ncnn found via CMake config")
else()
    if(NOT NCNN_ROOT)
        set(NCNN_ROOT $ENV{NCNN_ROOT})
    endif()

    set(_NCNN_SEARCH_PATHS
        ${NCNN_ROOT}
        /opt/homebrew
        /usr/local
        /usr
    )

    find_path(NCNN_INCLUDE_DIR
        NAMES net.h
        PATHS ${_NCNN_SEARCH_PATHS}
        PATH_SUFFIXES include/ncnn include)

    find_library(NCNN_LIB
        NAMES ncnn
        PATHS ${_NCNN_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64)

    if(NCNN_INCLUDE_DIR AND NCNN_LIB)
        set(NCNN_LINK_TARGET ${NCNN_LIB})
        message(STATUS "ncnn found: include=${NCNN_INCLUDE_DIR}, lib=${NCNN_LIB}")
    else()
        message(FATAL_ERROR
            "ncnn not found.\n"
            "Options:\n"
            "  1. Install via package manager (brew install ncnn)\n"
            "  2. Build from source: https://github.com/Tencent/ncnn#build\n"
            "  3. Specify path: cmake -DNCNN_ROOT=/path/to/ncnn ..\n"
            "  4. Set environment variable: export NCNN_ROOT=/path/to/ncnn")
    endif()
endif()

# ---- 查找 llama.cpp：优先 CMake config，失败则手动搜索 ----
find_package(llama QUIET)
if(llama_FOUND)
    set(LLAMA_LINK_TARGET llama)
    message(STATUS "llama.cpp found via CMake config")
    # llama CMake config 可能导出 ggml 依赖，尝试查找
    find_package(ggml QUIET)
    if(ggml_FOUND)
        set(GGML_LINK_TARGET ggml::ggml)
        message(STATUS "ggml found via CMake config")
    else()
        set(GGML_LINK_TARGET "")
        message(STATUS "ggml not found separately, assuming linked into llama")
    endif()
else()
    if(NOT LLAMA_ROOT)
        set(LLAMA_ROOT $ENV{LLAMA_ROOT})
    endif()

    set(_LLAMA_SEARCH_PATHS
        ${LLAMA_ROOT}
        /opt/homebrew
        /usr/local
        /usr
    )

    find_path(LLAMA_INCLUDE_DIR
        NAMES llama.h
        PATHS ${_LLAMA_SEARCH_PATHS}
        PATH_SUFFIXES include)

    find_library(LLAMA_LIB
        NAMES llama
        PATHS ${_LLAMA_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64)

    find_library(GGML_LIB
        NAMES ggml
        PATHS ${_LLAMA_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64)

    if(LLAMA_INCLUDE_DIR AND LLAMA_LIB)
        set(LLAMA_LINK_TARGET ${LLAMA_LIB})
        message(STATUS "llama.cpp found: include=${LLAMA_INCLUDE_DIR}, lib=${LLAMA_LIB}")
        if(GGML_LIB)
            set(GGML_LINK_TARGET ${GGML_LIB})
            message(STATUS "ggml found: lib=${GGML_LIB}")
        else()
            set(GGML_LINK_TARGET "")
            message(STATUS "ggml not found separately, assuming linked into llama")
        endif()
    else()
        message(FATAL_ERROR
            "llama.cpp not found.\n"
            "Options:\n"
            "  1. Install via package manager (brew install llama.cpp)\n"
            "  2. Build from source: https://github.com/ggml-org/llama.cpp\n"
            "  3. Specify path: cmake -DLLAMA_ROOT=/path/to/llama.cpp ..\n"
            "  4. Set environment variable: export LLAMA_ROOT=/path/to/llama.cpp")
    endif()
endif()

add_executable(inference_service
    src/main.cpp
    src/utils/logger.cpp
    src/utils/timer.cpp
    src/preprocess/preprocess.cpp
    src/backend/onnx_backend.cpp
    src/backend/ncnn_backend.cpp
    src/postprocess/yolo_postprocess.cpp
    src/detector/object_detector.cpp
    src/llm/llm_generator.cpp
    src/pipeline/batch_preprocessor.cpp
    src/server/http_server.cpp
)

target_include_directories(inference_service PRIVATE include)

# 手动搜索模式需要额外添加 onnxruntime include 目录
if(ONNXRUNTIME_INCLUDE_DIR)
    target_include_directories(inference_service PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
endif()

# 手动搜索模式需要额外添加 ncnn include 目录
if(NCNN_INCLUDE_DIR)
    target_include_directories(inference_service PRIVATE ${NCNN_INCLUDE_DIR})
endif()

# 手动搜索模式需要额外添加 llama include 目录
if(LLAMA_INCLUDE_DIR)
    target_include_directories(inference_service PRIVATE ${LLAMA_INCLUDE_DIR})
endif()

target_link_libraries(inference_service ${OpenCV_LIBS} ${ORT_LINK_TARGET} ${NCNN_LINK_TARGET} ${LLAMA_LINK_TARGET} ${GGML_LINK_TARGET} httplib::httplib)

# macOS 上 ncnn Vulkan 依赖 Metal 框架，手动搜索模式下可能需要显式链接
if(APPLE AND NOT ncnn_FOUND)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        target_link_libraries(inference_service ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
        message(STATUS "Linking Metal frameworks for ncnn Vulkan support")
    endif()
endif()

# ---- 单元测试（M10）----
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # 优先使用 Homebrew GTest（静态库），避免 anaconda3 GTest（动态库）ABI 不兼容
    # anaconda3 的 libgtest.dylib 链接 libc++，与 Apple Clang arm64 存在 ABI 差异导致链接失败
    if(NOT GTest_ROOT)
        if(APPLE AND EXISTS /opt/homebrew/lib/cmake/GTest)
            set(GTest_ROOT /opt/homebrew)
        endif()
    endif()

    find_package(GTest REQUIRED)

    # 公共测试源文件（非测试框架的模块源码）
    set(TEST_SOURCES
        src/utils/logger.cpp
        src/utils/timer.cpp
        src/preprocess/preprocess.cpp
        src/backend/onnx_backend.cpp
        src/backend/ncnn_backend.cpp
        src/postprocess/yolo_postprocess.cpp
        src/detector/object_detector.cpp
        src/llm/llm_generator.cpp
        src/pipeline/batch_preprocessor.cpp
        src/server/http_server.cpp
    )

    # 辅助函数：创建单个测试目标
    function(add_inference_test test_name)
        add_executable(${test_name} tests/${test_name}.cpp ${TEST_SOURCES})
        target_include_directories(${test_name} PRIVATE include)
        if(ONNXRUNTIME_INCLUDE_DIR)
            target_include_directories(${test_name} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        endif()
        if(NCNN_INCLUDE_DIR)
            target_include_directories(${test_name} PRIVATE ${NCNN_INCLUDE_DIR})
        endif()
        if(LLAMA_INCLUDE_DIR)
            target_include_directories(${test_name} PRIVATE ${LLAMA_INCLUDE_DIR})
        endif()
        target_link_libraries(${test_name}
            ${OpenCV_LIBS} ${ORT_LINK_TARGET} ${NCNN_LINK_TARGET}
            ${LLAMA_LINK_TARGET} ${GGML_LINK_TARGET} httplib::httplib
            GTest::gtest GTest::gtest_main)
        if(APPLE AND NOT ncnn_FOUND)
            find_library(METAL_FRAMEWORK_TEST Metal)
            find_library(FOUNDATION_FRAMEWORK_TEST Foundation)
            if(METAL_FRAMEWORK_TEST AND FOUNDATION_FRAMEWORK_TEST)
                target_link_libraries(${test_name} ${METAL_FRAMEWORK_TEST} ${FOUNDATION_FRAMEWORK_TEST})
            endif()
        endif()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

    add_inference_test(test_logger)
    add_inference_test(test_timer)
    add_inference_test(test_preprocess)
    add_inference_test(test_yolo_postprocess)
    add_inference_test(test_batch_preprocessor)
    add_inference_test(test_object_detector)
endif()

# ---- 性能基准测试（M11）----
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)

if(BUILD_BENCHMARKS)
    add_executable(performance_benchmark
        benchmarks/performance_benchmark.cpp
        src/utils/logger.cpp
        src/utils/timer.cpp
        src/preprocess/preprocess.cpp
        src/backend/onnx_backend.cpp
        src/backend/ncnn_backend.cpp
        src/postprocess/yolo_postprocess.cpp
        src/detector/object_detector.cpp
    )
    target_include_directories(performance_benchmark PRIVATE include)
    if(ONNXRUNTIME_INCLUDE_DIR)
        target_include_directories(performance_benchmark PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    endif()
    if(NCNN_INCLUDE_DIR)
        target_include_directories(performance_benchmark PRIVATE ${NCNN_INCLUDE_DIR})
    endif()
    target_link_libraries(performance_benchmark
        ${OpenCV_LIBS} ${ORT_LINK_TARGET} ${NCNN_LINK_TARGET})
    if(APPLE AND NOT ncnn_FOUND)
        find_library(METAL_FRAMEWORK_BENCH Metal)
        find_library(FOUNDATION_FRAMEWORK_BENCH Foundation)
        if(METAL_FRAMEWORK_BENCH AND FOUNDATION_FRAMEWORK_BENCH)
            target_link_libraries(performance_benchmark
                ${METAL_FRAMEWORK_BENCH} ${FOUNDATION_FRAMEWORK_BENCH})
        endif()
    endif()
endif()
