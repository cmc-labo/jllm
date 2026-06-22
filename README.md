# local-llm

A lightweight Java tool for managing and running local LLMs — similar to Ollama but minimal by design.

Models are registered by pointing to a local GGUF file and a [llama.cpp](https://github.com/ggerganov/llama.cpp) binary. The registry is stored in `~/.local-llm/models.json`.

## Requirements

- Java 11+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-cli` binary) for running models
- A GGUF model file

## Build

**Without Maven** (recommended — downloads Gson, SLF4J, and Logback automatically):

```bash
bash build.sh
```

**With Maven:**

```bash
mvn package
```

Output: `target/local-llm.jar`

## Usage

```
java -jar target/local-llm.jar <command> [options]
```

### Commands

| Command | Description |
|---|---|
| `list` | List registered models |
| `add <name> --path <path>` | Register a model |
| `rm <name>` | Remove a model |
| `run <name>` | Start an interactive chat session |
| `serve [--port <port>]` | Start the HTTP API server (default: 11434) |
| `info <name>` | Show model details |

### Options for `add`

| Flag | Description |
|---|---|
| `--path <path>` | **(required)** Path to the GGUF model file |
| `--binary <path>` | Path to `llama-cli`. Auto-detected if omitted |
| `--format <fmt>` | Model format (default: `gguf`) |

## Examples

```bash
JAR="java -jar target/local-llm.jar"

# Register a model
$JAR add phi3:mini --path ~/models/phi3-mini-4k-instruct-q4.gguf --binary /usr/local/bin/llama-cli

# List registered models
$JAR list

# Show model details
$JAR info phi3:mini

# Interactive chat
$JAR run phi3:mini

# Start API server on default port 11434
$JAR serve

# Start API server on a custom port
$JAR serve --port 8080

# Remove a model
$JAR rm phi3:mini
```

## HTTP API

The server exposes an Ollama-compatible REST API.

### `GET /api/tags` — List models

```bash
curl http://localhost:11434/api/tags
```

```json
{
  "models": [
    {
      "name": "phi3:mini",
      "size": 2394025984,
      "modified_at": "2026-06-16T00:00:00Z",
      "details": { "format": "gguf" }
    }
  ]
}
```

### `POST /api/generate` — Generate text

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "phi3:mini",
    "prompt": "Why is the sky blue?",
    "options": { "num_predict": 200 }
  }'
```

```json
{
  "model": "phi3:mini",
  "created_at": "2026-06-16T00:00:00Z",
  "response": "...",
  "done": true
}
```

### `POST /api/chat` — Chat completion

```bash
curl http://localhost:11434/api/chat \
  -d '{
    "model": "phi3:mini",
    "messages": [
      { "role": "user", "content": "Hello!" }
    ]
  }'
```

```json
{
  "model": "phi3:mini",
  "created_at": "2026-06-16T00:00:00Z",
  "message": { "role": "assistant", "content": "Hello! How can I help you?" },
  "done": true
}
```

## Project Structure

```
local-llm-env/
├── build.sh                              # Maven-free build script
├── pom.xml                               # Maven build file
├── native/                               # JNI wrapper around llama.cpp's C API
│   ├── CMakeLists.txt
│   ├── build.sh
│   └── llama_jni.cpp
├── src/main/resources/
│   └── logback.xml                       # Default Logback config (console, native logger at INFO)
└── src/main/java/dev/localllm/
    ├── Main.java                         # CLI entry point
    ├── model/
    │   ├── ModelConfig.java              # Model POJO
    │   └── ModelRegistry.java            # Persists registry to ~/.local-llm/models.json
    ├── runner/
    │   └── ModelRunner.java              # Runs llama.cpp as a subprocess
    ├── server/
    │   └── ApiServer.java                # Ollama-compatible HTTP API server
    └── jni/
        ├── LlamaNative.java              # Raw native method declarations
        ├── NativeLibraryLoader.java      # Locates and loads libllamajni.so
        ├── NativeLogBridge.java          # Forwards native log output into SLF4J
        ├── NativeCrashException.java     # Thrown when a native fatal signal is caught
        ├── LlamaModel.java               # High-level model wrapper (AutoCloseable)
        ├── LlamaContext.java             # High-level inference context wrapper
        └── LlamaDemo.java                # Minimal smoke-test CLI
```

## Notes

- The registry file lives at `~/.local-llm/models.json` and persists across sessions.
- The API server uses the JDK's built-in `com.sun.net.httpserver.HttpServer` — no extra HTTP framework dependency.
- Chat prompts are formatted using [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md), which is compatible with most modern GGUF models.
- Logging goes through SLF4J ([Logback](https://logback.qos.ch/) by default, see `src/main/resources/logback.xml`); this includes llama.cpp/ggml's own native log output (see [Native log output](#native-log-output) below).

## JNI Binding (`dev.localllm.jni`)

In addition to the subprocess-based CLI above, this project includes a direct **JNI binding to llama.cpp's C API**, so Java code can run inference in-process (no `llama-cli` subprocess, no stdout parsing).

### Why JNI

llama.cpp is a C/C++ library; running models fast on CPU/GPU requires linking against it directly rather than shelling out. This binding links a small C++ JNI shim (`native/llama_jni.cpp`) against `libllama.so`, exposing model loading, tokenization, and a streaming generation loop to Java via `dev.localllm.jni.LlamaNative`.

### Build steps

**1. Build llama.cpp as a shared library** (one-time; not vendored in this repo, see `.gitignore`):

```bash
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama ggml
cd ..
```

**2. Generate the JNI header and compile the Java classes:**

```bash
javac -h jni-headers -d target/classes src/main/java/dev/localllm/jni/*.java
```

**3. Build the JNI shared library** (links against `llama.cpp/build/bin/libllama.so`):

```bash
bash native/build.sh
```

This produces `native/build/libllamajni.so`, linked with an rpath back to `llama.cpp/build/bin` so `libllama.so` / `libggml*.so` resolve automatically at runtime.

### Usage from Java

```java
import dev.localllm.jni.LlamaModel;
import dev.localllm.jni.LlamaContext;

try (LlamaModel model = new LlamaModel("/path/to/model.gguf", /* nGpuLayers */ 0);
     LlamaContext ctx = model.createContext(/* nCtx */ 2048, /* nThreads */ 4)) {

    // One-shot:
    String text = ctx.generate("Once upon a time", /* nPredict */ 128, /* temperature */ 0.8f);

    // Streaming:
    ctx.generateStreaming("Once upon a time", 128, 0.8f, piece -> System.out.print(piece));
}
```

`LlamaNative` searches for `libllamajni.so` in this order: `-Ddev.localllm.nativeLib=<file>`, `-Ddev.localllm.nativeLibDir=<dir>`, or `./native/build/libllamajni.so` relative to the working directory.

### Smoke test

```bash
java -Ddev.localllm.nativeLibDir=native/build -cp target/classes \
  dev.localllm.jni.LlamaDemo /path/to/model.gguf "Once upon a time"
```

Verified end-to-end against [`ggml-org/models/tinyllamas/stories260K.gguf`](https://huggingface.co/ggml-org/models) (a 1.2 MB test model), producing coherent greedy-sampled output.

### API surface

| Class | Purpose |
|---|---|
| `LlamaNative` | Raw `native` method declarations mirroring llama.cpp's C API (model/context lifecycle, tokenize, detokenize, EOG check, streaming generate, log callback registration) |
| `LlamaModel` | `AutoCloseable` wrapper; loads a GGUF file, exposes tokenization helpers and `createContext()` |
| `LlamaContext` | `AutoCloseable` wrapper bound to a model; exposes `generate()` and `generateStreaming()` |
| `NativeLogBridge` | Forwards llama.cpp/ggml native log output into SLF4J (see [Native log output](#native-log-output)) |
| `NativeCrashException` | Thrown when native code hits a fatal signal that was intercepted instead of killing the JVM (see [Crash containment](#crash-containment)) |

Generation uses greedy sampling when `temperature <= 0`, otherwise temperature + distribution sampling (`llama_sampler_init_temp` + `llama_sampler_init_dist`).

### Concurrency

A single `llama_context` is not reentrant: concurrent `llama_decode` calls against it corrupt its KV cache / sampler state. `LlamaContext` guards every native call with a lock, so calls on a *shared* context are serialized rather than racing — but for actual parallel generation across users, create one `LlamaContext` per concurrent request via `LlamaModel.createContext()`. The underlying `LlamaModel` is safe to share across contexts (this matches llama.cpp's own multi-slot server design).

### Crash containment

Bad input or a native bug inside llama.cpp/ggml can otherwise either escape as a C++ exception across the JNI boundary (undefined behavior) or raise a fatal signal (`SIGSEGV`, `SIGABRT` from a failed `GGML_ASSERT`, `SIGBUS`, `SIGFPE`, `SIGILL`) that kills the JVM process outright. `native/llama_jni.cpp` guards against both: arguments are validated before touching native memory, C++ exceptions are caught and rethrown as Java exceptions, and a per-call signal handler converts a caught fatal signal into a `NativeCrashException` instead of taking down the process.

This is best-effort containment, not full recovery: a signal caught mid-call may leave native heap state corrupted in ways invisible from Java, so once any call has crashed, every subsequent native call in the process refuses to run (throwing `NativeCrashException` immediately) — restart the process rather than continuing to use it. It also only covers the JNI-calling thread; a crash on one of ggml's internal worker threads (batched decode with `nThreads > 1`) is not caught.

### Native log output

llama.cpp/ggml log everything (model loading, hardware/backend detection, decode warnings, the tensor-loading progress dots) through `llama_log_set`/`ggml_log_set` rather than printing directly. `LlamaModel` registers `NativeLogBridge` before `LlamaNative.backendInit()`, which forwards every native log line into the SLF4J logger named `dev.localllm.native` (mapped to `debug`/`info`/`warn`/`error`; multi-part "continuation" output like the progress dots is coalesced into a single line first). Control verbosity the normal SLF4J/Logback way, e.g. in your own `logback.xml`:

```xml
<logger name="dev.localllm.native" level="WARN"/>
```

or override the bundled default entirely with `-Dlogback.configurationFile=/path/to/logback.xml`.
