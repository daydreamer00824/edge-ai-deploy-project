cmake_minimum_required(VERSION 3.22)

project(test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# =========================
# ONNX Runtime
# =========================
set(ORT_DIR /home/daydreamer/third_party/onnxruntime)
set(ORT_INCLUDE_DIR ${ORT_DIR}/include)
set(ORT_LIB_DIR ${ORT_DIR}/lib)
set(ORT_LIB ${ORT_LIB_DIR}/libonnxruntime.so)

# =========================
# TensorRT
# =========================
find_path(TRT_INCLUDE_DIR NvInfer.h
        HINTS
        /usr/include/x86_64-linux-gnu
        /usr/include
        /usr/local/TensorRT/include
)

find_library(TRT_NVINFER_LIB nvinfer
        HINTS
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/TensorRT/lib
        /usr/local/TensorRT/lib64
)

if(NOT TRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT include path not found. NvInfer.h missing.")
endif()

if(NOT TRT_NVINFER_LIB)
    message(FATAL_ERROR "TensorRT nvinfer library not found. libnvinfer.so missing.")
endif()

message(STATUS "TensorRT include dir: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer lib: ${TRT_NVINFER_LIB}")

get_filename_component(TRT_LIB_DIR ${TRT_NVINFER_LIB} DIRECTORY)

# =========================
# ONNX Runtime executables
# =========================
add_executable(cpp_ort_infer
        src/main.cpp
        src/timer.cpp
        src/image_process.cpp
        src/postprocess.cpp
        src/label_map.cpp
        src/ort_runner.cpp
        src/batch_infer.cpp
)

add_executable(cpp_ort_benchmark
        src/benchmark_main.cpp
        src/benchmark.cpp
        src/timer.cpp
        src/image_process.cpp
        src/postprocess.cpp
        src/label_map.cpp
        src/ort_runner.cpp
        src/batch_infer.cpp
)

add_executable(verify_cpp_ort
        src/verify_cpp_ort.cpp
        src/image_process.cpp
        src/postprocess.cpp
        src/label_map.cpp
        src/ort_runner.cpp
)

add_executable(verify_ort_cpu_cuda_batch
        src/verify_ort_cpu_cuda_batch.cpp
        src/image_process.cpp
        src/ort_runner.cpp
        src/batch_infer.cpp
)

foreach(target
        cpp_ort_infer
        cpp_ort_benchmark
        verify_cpp_ort
        verify_ort_cpu_cuda_batch
)
    target_include_directories(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${ORT_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(${target} PRIVATE
            ${OpenCV_LIBS}
            ${ORT_LIB}
    )

    set_target_properties(${target} PROPERTIES
            BUILD_RPATH ${ORT_LIB_DIR}
            INSTALL_RPATH ${ORT_LIB_DIR}
    )
endforeach()

# =========================
# TensorRT executables
# =========================
add_executable(cpp_trt_infer
        src/trt_infer_main.cpp
        src/trt_runner.cpp
        src/timer.cpp
        src/image_process.cpp
        src/postprocess.cpp
        src/label_map.cpp
        src/batch_infer.cpp
)

add_executable(cpp_trt_benchmark
        src/trt_benchmark_main.cpp
        src/trt_runner.cpp
        src/benchmark.cpp
        src/timer.cpp
        src/image_process.cpp
        src/postprocess.cpp
        src/label_map.cpp
        src/batch_infer.cpp
)

add_executable(verify_trt_buffer_reuse
        src/verify_trt_buffer_reuse.cpp
        src/trt_runner.cpp
        src/image_process.cpp
        src/batch_infer.cpp
)

foreach(target
        cpp_trt_infer
        cpp_trt_benchmark
        verify_trt_buffer_reuse
)
    target_include_directories(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${TRT_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(${target} PRIVATE
            ${OpenCV_LIBS}
            ${TRT_NVINFER_LIB}
            CUDA::cudart
    )

    set_target_properties(${target} PROPERTIES
            BUILD_RPATH ${TRT_LIB_DIR}
            INSTALL_RPATH ${TRT_LIB_DIR}
    )
endforeach()
