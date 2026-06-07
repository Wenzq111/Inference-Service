cmake_minimum_required(VERSION 3.15)
project(Inference-Service VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# 查找 onnxruntime：优先使用 CMake imported target，失败则手动搜索 Homebrew 路径
find_package(onnxruntime QUIET)
if(onnxruntime_FOUND)
    # imported target onnxruntime::onnxruntime 自动提供 include 和 lib
    set(ORT_LINK_TARGET onnxruntime::onnxruntime)
    message(STATUS "onnxruntime found via CMake config")
else()
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS /opt/homebrew/include/onnxruntime
              /usr/local/include/onnxruntime)
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS /opt/homebrew/lib
              /usr/local/lib)
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        set(ORT_LINK_TARGET ${ONNXRUNTIME_LIB})
        message(STATUS "onnxruntime found manually: ${ONNXRUNTIME_LIB}")
    else()
        message(FATAL_ERROR "onnxruntime not found. Install via: brew install onnxruntime")
    endif()
endif()

add_executable(inference_service
    src/main.cpp
    src/utils/logger.cpp
    src/utils/timer.cpp
    src/preprocess/preprocess.cpp
    src/backend/onnx_backend.cpp
)

target_include_directories(inference_service PRIVATE include)

# 手动搜索模式需要额外添加 onnxruntime include 目录
if(ONNXRUNTIME_INCLUDE_DIR)
    target_include_directories(inference_service PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
endif()

target_link_libraries(inference_service ${OpenCV_LIBS} ${ORT_LINK_TARGET})