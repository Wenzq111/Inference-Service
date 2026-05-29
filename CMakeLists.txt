cmake_minimum_required(VERSION 3.15)
project(Inference-Service VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(inference_service
    src/main.cpp
    src/utils/logger.cpp
    src/utils/timer.cpp
)

target_include_directories(inference_service PRIVATE include)