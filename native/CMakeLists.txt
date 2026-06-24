cmake_minimum_required(VERSION 3.16)
project(llamajni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# When ON, llama.cpp is compiled from source (as a cmake subdirectory) and
# statically linked into libllamajni, producing a single self-contained
# shared library suitable for bundling inside a JAR.
#
# When OFF (default), links dynamically against a pre-built libllama.so /
# libggml*.so in LLAMA_CPP_DIR/build/bin - the fast path for local dev.
option(LLAMAJNI_STATIC
    "Build llama.cpp from source and link it statically into libllamajni"
    OFF)

find_package(JNI REQUIRED)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp"
    CACHE PATH "Path to the llama.cpp source checkout")
set(JNI_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jni-headers"
    CACHE PATH "Directory containing the javac -h generated JNI header")

add_library(llamajni SHARED llama_jni.cpp)

target_include_directories(llamajni PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${JNI_HEADERS_DIR}
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
)

if(LLAMAJNI_STATIC)
    # Build llama.cpp from source; force all its sub-targets to be static
    # archives so they get embedded into libllamajni.so.  libllamajni itself
    # remains a SHARED library (the JVM can only dlopen shared libraries),
    # but has no runtime dependency on a separately installed libllama.so.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp sub-targets as static archives" FORCE)

    # Suppress llama.cpp's own tools/tests/server - we only need the libs.
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

    add_subdirectory(${LLAMA_CPP_DIR} llama_build EXCLUDE_FROM_ALL)

    target_link_libraries(llamajni PRIVATE llama ggml)

    if(UNIX AND NOT APPLE)
        # pthreads and dl are referenced by ggml's thread pool and dlopen-based
        # backend loading; they must be provided at link time for static builds.
        target_link_libraries(llamajni PRIVATE pthread dl)
    endif()

    # No RPATH: the library is self-contained, nothing to find at runtime
    # (other than system libraries that are guaranteed to be present).
    set_target_properties(llamajni PROPERTIES
        INSTALL_RPATH "")
else()
    # Default (dev mode): link dynamically against a pre-built llama.cpp.
    # Requires ../llama.cpp/build/bin/libllama.so and libggml*.so to exist.
    target_link_directories(llamajni PRIVATE ${LLAMA_CPP_DIR}/build/bin)
    target_link_libraries(llamajni PRIVATE llama ggml)

    # Resolve libllama.so / libggml*.so at runtime relative to this library's
    # own location, so callers only need to load libllamajni.so by path.
    set_target_properties(llamajni PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN:${LLAMA_CPP_DIR}/build/bin")
endif()
