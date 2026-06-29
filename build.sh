#!/bin/bash
# Build script — no Maven required. Downloads dependency JARs and produces a fat JAR.
set -e

GSON_VERSION="2.10.1"
GSON_JAR="lib/gson-${GSON_VERSION}.jar"
GSON_URL="https://repo1.maven.org/maven2/com/google/code/gson/gson/${GSON_VERSION}/gson-${GSON_VERSION}.jar"

SLF4J_VERSION="2.0.13"
SLF4J_JAR="lib/slf4j-api-${SLF4J_VERSION}.jar"
SLF4J_URL="https://repo1.maven.org/maven2/org/slf4j/slf4j-api/${SLF4J_VERSION}/slf4j-api-${SLF4J_VERSION}.jar"

LOGBACK_VERSION="1.5.6"
LOGBACK_CLASSIC_JAR="lib/logback-classic-${LOGBACK_VERSION}.jar"
LOGBACK_CLASSIC_URL="https://repo1.maven.org/maven2/ch/qos/logback/logback-classic/${LOGBACK_VERSION}/logback-classic-${LOGBACK_VERSION}.jar"
LOGBACK_CORE_JAR="lib/logback-core-${LOGBACK_VERSION}.jar"
LOGBACK_CORE_URL="https://repo1.maven.org/maven2/ch/qos/logback/logback-core/${LOGBACK_VERSION}/logback-core-${LOGBACK_VERSION}.jar"

UNDERTOW_VERSION="2.3.14.Final"
UNDERTOW_JAR="lib/undertow-core-${UNDERTOW_VERSION}.jar"
UNDERTOW_URL="https://repo1.maven.org/maven2/io/undertow/undertow-core/${UNDERTOW_VERSION}/undertow-core-${UNDERTOW_VERSION}.jar"

XNIO_VERSION="3.8.14.Final"
XNIO_API_JAR="lib/xnio-api-${XNIO_VERSION}.jar"
XNIO_API_URL="https://repo1.maven.org/maven2/org/jboss/xnio/xnio-api/${XNIO_VERSION}/xnio-api-${XNIO_VERSION}.jar"
XNIO_NIO_JAR="lib/xnio-nio-${XNIO_VERSION}.jar"
XNIO_NIO_URL="https://repo1.maven.org/maven2/org/jboss/xnio/xnio-nio/${XNIO_VERSION}/xnio-nio-${XNIO_VERSION}.jar"

JBOSS_LOGGING_VERSION="3.5.3.Final"
JBOSS_LOGGING_JAR="lib/jboss-logging-${JBOSS_LOGGING_VERSION}.jar"
JBOSS_LOGGING_URL="https://repo1.maven.org/maven2/org/jboss/logging/jboss-logging/${JBOSS_LOGGING_VERSION}/jboss-logging-${JBOSS_LOGGING_VERSION}.jar"

WILDFLY_COMMON_VERSION="1.7.0.Final"
WILDFLY_COMMON_JAR="lib/wildfly-common-${WILDFLY_COMMON_VERSION}.jar"
WILDFLY_COMMON_URL="https://repo1.maven.org/maven2/org/wildfly/common/wildfly-common/${WILDFLY_COMMON_VERSION}/wildfly-common-${WILDFLY_COMMON_VERSION}.jar"

JBOSS_THREADS_VERSION="3.5.0.Final"
JBOSS_THREADS_JAR="lib/jboss-threads-${JBOSS_THREADS_VERSION}.jar"
JBOSS_THREADS_URL="https://repo1.maven.org/maven2/org/jboss/threads/jboss-threads/${JBOSS_THREADS_VERSION}/jboss-threads-${JBOSS_THREADS_VERSION}.jar"

OUT_JAR="target/local-llm.jar"

echo "=== local-llm build ==="

mkdir -p lib target/classes target/fat

download_if_missing() {
    local dest="$1" url="$2"
    if [ ! -f "$dest" ]; then
        echo "Downloading $(basename "$dest")..."
        curl -fsSL -o "$dest" "$url"
    fi
}

download_if_missing "$GSON_JAR"           "$GSON_URL"
download_if_missing "$SLF4J_JAR"          "$SLF4J_URL"
download_if_missing "$LOGBACK_CLASSIC_JAR" "$LOGBACK_CLASSIC_URL"
download_if_missing "$LOGBACK_CORE_JAR"   "$LOGBACK_CORE_URL"
download_if_missing "$UNDERTOW_JAR"       "$UNDERTOW_URL"
download_if_missing "$XNIO_API_JAR"       "$XNIO_API_URL"
download_if_missing "$XNIO_NIO_JAR"       "$XNIO_NIO_URL"
download_if_missing "$JBOSS_LOGGING_JAR"  "$JBOSS_LOGGING_URL"
download_if_missing "$WILDFLY_COMMON_JAR" "$WILDFLY_COMMON_URL"
download_if_missing "$JBOSS_THREADS_JAR" "$JBOSS_THREADS_URL"

CP="$GSON_JAR:$SLF4J_JAR:$LOGBACK_CLASSIC_JAR:$LOGBACK_CORE_JAR:\
$UNDERTOW_JAR:$XNIO_API_JAR:$XNIO_NIO_JAR:$JBOSS_LOGGING_JAR:$WILDFLY_COMMON_JAR:$JBOSS_THREADS_JAR"

echo "Compiling..."
find src/main/java -name "*.java" | sort > /tmp/local_llm_sources.txt
javac -source 11 -target 11 -cp "$CP" -d target/classes @/tmp/local_llm_sources.txt
rm /tmp/local_llm_sources.txt

if [ -d src/main/resources ]; then
    cp -r src/main/resources/* target/classes/
fi

# Bundle any pre-built native libraries for JAR-based auto-loading.
# Build them first with: bash native/build.sh --static [--variant cuda|rocm|metal]
# Each variant goes into native/dist/<classifier>/ and is bundled at
# native/<classifier>/<filename> inside the JAR so NativeLibraryLoader can
# extract and load the right one at runtime.
if [ -d native/dist ] && [ -n "$(ls -A native/dist 2>/dev/null)" ]; then
    echo "Bundling native libraries from native/dist/..."
    mkdir -p target/classes/native
    cp -r native/dist/* target/classes/native/
    for classifier in native/dist/*/; do
        echo "  + $(basename "$classifier")"
    done
fi

echo "Packaging fat JAR..."
cp -r target/classes/* target/fat/
cd target/fat
jar xf "../../$GSON_JAR"
jar xf "../../$SLF4J_JAR"
jar xf "../../$LOGBACK_CORE_JAR"
jar xf "../../$LOGBACK_CLASSIC_JAR"
jar xf "../../$UNDERTOW_JAR"
jar xf "../../$XNIO_API_JAR"
jar xf "../../$XNIO_NIO_JAR"
jar xf "../../$JBOSS_LOGGING_JAR"
jar xf "../../$WILDFLY_COMMON_JAR"
jar xf "../../$JBOSS_THREADS_JAR"
# Keep META-INF/services (SLF4J binding discovery relies on it) but drop
# the rest (per-jar manifests, module info, license files, ...) so they
# don't collide when multiple dependency JARs are merged into one.
find META-INF -mindepth 1 -maxdepth 1 ! -name services -exec rm -rf {} +
cd ../..

jar cfe "$OUT_JAR" dev.localllm.Main -C target/fat .

echo ""
echo "Build complete: $OUT_JAR"
echo ""
echo "Quick start:"
echo "  java -jar $OUT_JAR list"
echo "  java -jar $OUT_JAR add mymodel --path /path/to/model.gguf --binary /usr/local/bin/llama-cli"
echo "  java -jar $OUT_JAR run mymodel"
echo "  java -jar $OUT_JAR serve"
