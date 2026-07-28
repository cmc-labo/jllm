#!/bin/bash
# Runs the JUnit test suite — no Maven required.
#
# Prerequisite: run ./build.sh first (this script reuses target/classes and
# the dependency JARs it downloaded into lib/).
set -e

JUNIT_VERSION="1.10.2"
JUNIT_JAR="lib/junit-platform-console-standalone-${JUNIT_VERSION}.jar"
JUNIT_URL="https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/${JUNIT_VERSION}/junit-platform-console-standalone-${JUNIT_VERSION}.jar"

if [ ! -d target/classes ] || [ -z "$(ls -A target/classes 2>/dev/null)" ]; then
    echo "target/classes not found — run ./build.sh first." >&2
    exit 1
fi

echo "=== local-llm tests ==="

mkdir -p lib target/test-classes

if [ ! -f "$JUNIT_JAR" ]; then
    echo "Downloading junit-platform-console-standalone..."
    curl -fsSL -o "$JUNIT_JAR" "$JUNIT_URL"
fi

# Every JAR build.sh downloaded, minus the JUnit launcher itself.
MAIN_LIBS=$(ls lib/*.jar | grep -v "junit-platform-console-standalone" | tr '\n' ':')
CP="target/classes:${MAIN_LIBS}${JUNIT_JAR}"

echo "Compiling tests..."
find src/test/java -name "*.java" | sort > /tmp/local_llm_test_sources.txt
javac -source 11 -target 11 -cp "$CP" -d target/test-classes @/tmp/local_llm_test_sources.txt
rm /tmp/local_llm_test_sources.txt

echo "Running tests..."
java -jar "$JUNIT_JAR" execute \
    --classpath "target/test-classes:$CP" \
    --scan-classpath target/test-classes \
    --details=tree \
    --fail-if-no-tests
