cmake_minimum_required(VERSION 3.16)
project(llamajni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(JNI REQUIRED)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH "Path to llama.cpp checkout")
set(JNI_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jni-headers" CACHE PATH "Path to javac -h generated headers")

add_library(llamajni SHARED llama_jni.cpp)

target_include_directories(llamajni PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${JNI_HEADERS_DIR}
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
)

target_link_directories(llamajni PRIVATE ${LLAMA_CPP_DIR}/build/bin)
target_link_libraries(llamajni PRIVATE llama ggml)

# Resolve libllama.so / libggml*.so at runtime relative to this library's
# own location, so callers only need to load libllamajni.so by path.
set_target_properties(llamajni PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN:${LLAMA_CPP_DIR}/build/bin"
)
