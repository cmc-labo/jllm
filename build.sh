#!/bin/bash
# Maven不要のビルドスクリプト。Gsonを自動ダウンロードしてfat JARを生成する。
set -e

GSON_VERSION="2.10.1"
GSON_JAR="lib/gson-${GSON_VERSION}.jar"
GSON_URL="https://repo1.maven.org/maven2/com/google/code/gson/gson/${GSON_VERSION}/gson-${GSON_VERSION}.jar"
OUT_JAR="target/local-llm.jar"

echo "=== local-llm build ==="

mkdir -p lib target/classes target/fat

if [ ! -f "$GSON_JAR" ]; then
    echo "Downloading Gson ${GSON_VERSION}..."
    curl -fsSL -o "$GSON_JAR" "$GSON_URL"
    echo "Done."
fi

echo "Compiling..."
find src/main/java -name "*.java" | sort > /tmp/local_llm_sources.txt
javac -source 11 -target 11 -cp "$GSON_JAR" -d target/classes @/tmp/local_llm_sources.txt
rm /tmp/local_llm_sources.txt

echo "Packaging fat JAR..."
cp -r target/classes/* target/fat/
cd target/fat
jar xf "../../$GSON_JAR"
rm -rf META-INF
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
