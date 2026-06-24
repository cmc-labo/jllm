#!/bin/bash
# Build libllamajni (the JNI wrapper around llama.cpp) for one platform variant.
#
# Default (no flags): dynamic dev build against a pre-built llama.cpp checkout.
#   Required: ../llama.cpp/build/bin/libllama.so (BUILD_SHARED_LIBS=ON build).
#   Output:   native/build/libllamajni.so  (dev path, loaded by NativeLibraryLoader).
#
# --static: embed llama.cpp statically, producing a self-contained shared
#   library suitable for bundling inside the fat JAR.
#   Required: ../llama.cpp source checkout only (this script builds it too).
#   Output:   native/dist/<classifier>/<filename>
#
# Options:
#   --static          Link llama.cpp + ggml statically into libllamajni.
#   --variant <name>  GPU variant: cpu (default), cuda, rocm, metal.
#
# Examples:
#   bash native/build.sh                          # dev dynamic build
#   bash native/build.sh --static                 # CPU-only distribution build
#   bash native/build.sh --static --variant cuda  # CUDA distribution build
#
# The resulting static builds go into native/dist/<classifier>/.
# Run build.sh from the repo root to bundle them into the fat JAR:
#   bash build.sh
set -e

cd "$(dirname "$0")"

STATIC=false
VARIANT=cpu

while [ $# -gt 0 ]; do
    case "$1" in
        --static)           STATIC=true ;;
        --variant)          VARIANT="$2"; shift ;;
        --variant=*)        VARIANT="${1#--variant=}" ;;
        -h|--help)
            sed -n '/^#/{ s/^# \?//; p }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Detect platform classifier ──────────────────────────────────────────────

RAW_OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$RAW_OS" in
    linux*)  OS=linux   EXT=so ;;
    darwin*) OS=osx     EXT=dylib ;;
    msys*|cygwin*|mingw*) OS=windows EXT=dll ;;
    *) echo "Unsupported OS: $RAW_OS" >&2; exit 1 ;;
esac

RAW_ARCH=$(uname -m)
case "$RAW_ARCH" in
    arm64|aarch64) ARCH=aarch64 ;;
    x86_64|amd64)  ARCH=x86_64 ;;
    *) echo "Unsupported arch: $RAW_ARCH" >&2; exit 1 ;;
esac

BASE="${OS}-${ARCH}"
if [ "$VARIANT" = "cpu" ] || [ -z "$VARIANT" ]; then
    CLASSIFIER="$BASE"
else
    CLASSIFIER="${BASE}-${VARIANT}"
fi

# ── Map variant to CMake flags ───────────────────────────────────────────────

case "$VARIANT" in
    cuda)  GPU_FLAGS="-DGGML_CUDA=ON" ;;
    rocm)  GPU_FLAGS="-DGGML_HIP=ON" ;;
    metal) GPU_FLAGS="-DGGML_METAL=ON" ;;
    cpu|"") GPU_FLAGS="-DGGML_CUDA=OFF -DGGML_HIP=OFF" ;;
    *) echo "Unknown variant '$VARIANT'. Use: cpu, cuda, rocm, metal" >&2; exit 1 ;;
esac

# ── Check prerequisites ──────────────────────────────────────────────────────

if [ ! -f ../jni-headers/dev_localllm_jni_LlamaNative.h ]; then
    echo "error: JNI header not found. Generate it first:" >&2
    echo "  javac -h ../jni-headers -d ../target/classes \\" >&2
    echo "      ../src/main/java/dev/localllm/jni/*.java" >&2
    exit 1
fi

# ── Build ─────────────────────────────────────────────────────────────────────

if $STATIC; then
    # ── Static / distribution build ──────────────────────────────────────────
    # Builds llama.cpp from source as part of the cmake build and links it
    # all statically into a self-contained libllamajni.so / .dylib / .dll.
    OUT_DIR="$(pwd)/dist/${CLASSIFIER}"
    mkdir -p "$OUT_DIR"
    BUILD_DIR="$(pwd)/build-${CLASSIFIER}"

    echo "=== Static build: ${CLASSIFIER} ==="
    cmake -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMAJNI_STATIC=ON \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$OUT_DIR" \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$OUT_DIR" \
        $GPU_FLAGS \
        "$(pwd)"
    cmake --build "$BUILD_DIR" --target llamajni -j "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

    echo ""
    echo "Static build complete: ${OUT_DIR}/libllamajni.${EXT}"
    echo "Bundle in JAR by running: bash build.sh"
else
    # ── Dynamic / dev build ───────────────────────────────────────────────────
    if [ ! -f ../llama.cpp/build/bin/libllama.so ] && \
       [ ! -f ../llama.cpp/build/bin/libllama.dylib ]; then
        echo "error: pre-built llama.cpp not found." >&2
        echo "Build it first (shared libs, for dev use):" >&2
        echo "  cd ../llama.cpp" >&2
        echo "  cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release" >&2
        echo "  cmake --build build -j\$(nproc)" >&2
        echo "" >&2
        echo "Or use --static to build everything from source in one step." >&2
        exit 1
    fi

    BUILD_DIR="$(pwd)/build"
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $GPU_FLAGS "$(pwd)"
    cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

    echo ""
    echo "Dev build complete: native/build/libllamajni.${EXT}"
fi
