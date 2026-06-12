cmake_minimum_required(VERSION 3.15)
project(Inference-Service VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(inference_service
    src/main.cpp
    src/utils/logger.cpp
    src/utils/timer.cpp
    src/preprocess/preprocess.cpp
    src/backend/onnx_backend.cpp
    src/backend/ncnn_backend.cpp
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

target_link_libraries(inference_service ${OpenCV_LIBS} ${ORT_LINK_TARGET} ${NCNN_LINK_TARGET})

# macOS 上 ncnn Vulkan 依赖 Metal 框架，手动搜索模式下可能需要显式链接
if(APPLE AND NOT ncnn_FOUND)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        target_link_libraries(inference_service ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
        message(STATUS "Linking Metal frameworks for ncnn Vulkan support")
    endif()
endif()
