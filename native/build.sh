#!/bin/bash
# Builds libllamajni.so against the llama.cpp checkout in ../llama.cpp.
# Requires: cmake, a C++17 compiler, a JDK (for jni.h), and a pre-built
# ../llama.cpp/build with BUILD_SHARED_LIBS=ON.
set -e

cd "$(dirname "$0")"

if [ ! -f ../llama.cpp/build/bin/libllama.so ]; then
    echo "error: ../llama.cpp/build/bin/libllama.so not found." >&2
    echo "Build llama.cpp first, e.g.:" >&2
    echo "  cd ../llama.cpp && cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
    exit 1
fi

if [ ! -f ../jni-headers/dev_localllm_jni_LlamaNative.h ]; then
    echo "error: ../jni-headers/dev_localllm_jni_LlamaNative.h not found." >&2
    echo "Generate it first, e.g.:" >&2
    echo "  javac -h ../jni-headers -d ../target/classes ../src/main/java/dev/localllm/jni/LlamaNative.java" >&2
    exit 1
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"

echo ""
echo "Build complete: native/build/libllamajni.so"
