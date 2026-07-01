#!/bin/bash
# Build the echo-tool plugin JAR.
# Run from this directory: bash build.sh /path/to/local-llm.jar
set -e

JAR="${1:-../../../target/local-llm.jar}"
if [ ! -f "$JAR" ]; then
    echo "Error: local-llm.jar not found at $JAR"
    echo "Usage: bash build.sh [/path/to/local-llm.jar]"
    exit 1
fi

javac -cp "$JAR" EchoTool.java
jar cf echo-tool.jar EchoTool.class META-INF/

echo "Built: echo-tool.jar"
echo "Install with:"
echo "  mkdir -p ~/.local-llm/plugins/"
echo "  cp echo-tool.jar ~/.local-llm/plugins/"
