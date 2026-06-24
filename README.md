# local-llm

A lightweight Java tool for managing and running local LLMs — similar to Ollama but minimal by design.

Models are registered by pointing to a local GGUF file and (optionally) a [llama.cpp](https://github.com/ggerganov/llama.cpp) binary. The registry is stored in `~/.local-llm/models.json`.

`run` (interactive chat) shells out to a `llama-cli` binary. `serve` (the HTTP API server) instead runs inference in-process via a JNI binding to llama.cpp - see [JNI Binding](#jni-binding-devlocalllmjni) - so it needs `native/build/libllamajni.so` built rather than a `llama-cli` binary.

## Requirements

- Java 11+
- A GGUF model file
- For `run`: [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-cli` binary)
- For `serve`: a built `libllamajni.so` (see [Build steps](#build-steps) under JNI Binding)

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

The server exposes an Ollama-compatible REST API. `/api/generate` and `/api/chat` stream by
default: the response is newline-delimited JSON (`application/x-ndjson`), one object per
generated token, flushed as soon as the JNI layer produces it - not buffered until generation
finishes. Pass `"stream": false` to instead get a single JSON object once generation completes.

The model named in `model` is loaded into memory on first use and kept resident for subsequent
requests (no reload-per-request); each request gets its own short-lived inference context.

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
    "options": { "num_predict": 200, "temperature": 0.8 }
  }'
```

Streaming response (default), one JSON object per line:

```json
{"model":"phi3:mini","created_at":"2026-06-16T00:00:00Z","response":"The","done":false}
{"model":"phi3:mini","created_at":"2026-06-16T00:00:00Z","response":" sky","done":false}
...
{"model":"phi3:mini","created_at":"2026-06-16T00:00:00Z","response":"","done":true}
```

With `"stream": false`:

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

Streaming response (default), one JSON object per line:

```json
{"model":"phi3:mini","created_at":"2026-06-16T00:00:00Z","message":{"role":"assistant","content":"Hello"},"done":false}
{"model":"phi3:mini","created_at":"2026-06-16T00:00:00Z","message":{"role":"assistant","content":"!"},"done":false}
...
{"model":"phi3:mini","created_at":"2026-06-16T00:00:00Z","message":{"role":"assistant","content":""},"done":true}
```

With `"stream": false`:

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
├── native/dist/                               # Pre-built native libs for JAR bundling
│   ├── linux-x86_64/libllamajni.so          #   (produced by native/build.sh --static)
│   ├── linux-x86_64-cuda/libllamajni.so
│   ├── osx-aarch64/libllamajni.dylib
│   └── windows-x86_64/llamajni.dll
├── src/main/resources/
│   └── logback.xml                       # Default Logback config (console, native logger at INFO)
└── src/main/java/dev/localllm/
    ├── Main.java                         # CLI entry point
    ├── model/
    │   ├── ModelConfig.java              # Model POJO
    │   └── ModelRegistry.java            # Persists registry to ~/.local-llm/models.json
    ├── runner/
    │   └── ModelRunner.java              # Runs llama.cpp as a subprocess (used by `run`)
    ├── server/
    │   └── ApiServer.java                # Ollama-compatible HTTP API server (JNI, in-process, streaming)
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

There are two build modes:

**Dev build** (fast, for local development):

```bash
# 1. Build llama.cpp as shared libs (once per checkout)
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && cd ..

# 2. Generate the JNI header
javac -h jni-headers -d target/classes src/main/java/dev/localllm/jni/*.java

# 3. Build the JNI wrapper (links dynamically against step 1's libllama.so)
bash native/build.sh
# → native/build/libllamajni.so  (uses rpath; libllama.so must stay nearby)
```

**Distribution / JAR-bundling build** (self-contained, for shipping):

Each platform variant is built separately and dropped into `native/dist/` where
`build.sh` picks them up automatically:

```bash
# Same prerequisite: llama.cpp source checkout at ./llama.cpp

# Generate JNI header (same as above)
javac -h jni-headers -d target/classes src/main/java/dev/localllm/jni/*.java

# CPU-only build for the current platform
bash native/build.sh --static
# → native/dist/linux-x86_64/libllamajni.so  (no external deps)

# CUDA build (needs CUDA toolkit, NVIDIA GPU)
bash native/build.sh --static --variant cuda
# → native/dist/linux-x86_64-cuda/libllamajni.so

# ROCm build (needs ROCm stack, AMD GPU)
bash native/build.sh --static --variant rocm
# → native/dist/linux-x86_64-rocm/libllamajni.so

# Package everything into the fat JAR (bundles all dist/ variants found)
bash build.sh
# → target/local-llm.jar  (contains native/linux-x86_64/... etc. as resources)
```

Run `bash native/build.sh --help` for the full option list.

### Native library loading and platform support

`NativeLibraryLoader` resolves the right `.so` / `.dylib` / `.dll` automatically at
runtime, in this priority order:

| Priority | Mechanism | When to use |
|---|---|---|
| 1 | `-Ddev.localllm.nativeLib=<absolute-path>` | Point to any single file |
| 2 | `-Ddev.localllm.nativeLibDir=<dir>` | Directory of pre-built libs |
| 3 | `./native/build/<filename>` (relative to CWD) | After a dev `native/build.sh` run |
| 4 | Extracted from inside the JAR | Fat-JAR distribution (no extra setup) |

For JAR-based loading (priority 4), the library is extracted to
`${java.io.tmpdir}/local-llm-native/<version>/<classifier>/` on first run and
reused on subsequent runs.

**Classifier naming:** `{os}-{arch}[-{gpu}]`

| os | arch | gpu variants |
|---|---|---|
| `linux` | `x86_64`, `aarch64` | `-cuda`, `-rocm` |
| `osx` | `x86_64`, `aarch64` | `-metal` (arm64 only; otherwise implicit) |
| `windows` | `x86_64` | `-cuda` |

**GPU auto-detection:** on Linux, CUDA presence is checked via
`/proc/driver/nvidia/version` (created by the NVIDIA kernel module); ROCm via
`/dev/kfd`. On Windows, `%SystemRoot%\System32\nvcuda.dll`. GPU-accelerated
variants are tried first; the CPU-only classifier is the final fallback. If a
detected GPU variant is not bundled in the JAR the next candidate is tried
automatically, so shipping CPU-only is always sufficient.

### Usage from Java

```java
import dev.localllm.jni.LlamaModel;
import dev.localllm.jni.LlamaContext;

try (LlamaModel model = new LlamaModel("/path/to/model.gguf", /* nGpuLayers */ 0);
     LlamaContext ctx = model.createContext(/* nCtx */ 2048, /* nThreads */ 4)) {

    // One-shot:
    String text = ctx.generate("Once upon a time", /* nPredict */ 128, /* temperature */ 0.8f);

    // Streaming via push callback:
    ctx.generateStreaming("Once upon a time", 128, 0.8f, piece -> System.out.print(piece));

    // Streaming via pull-based Iterator/Iterable - e.g. to write each token
    // straight to an HTTP response as it's produced (see ApiServer):
    try (LlamaContext.TokenStream tokens = ctx.generateTokens("Once upon a time", 128, 0.8f)) {
        for (String piece : tokens) {
            System.out.print(piece);
        }
    }
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
| `LlamaContext` | `AutoCloseable` wrapper bound to a model; exposes `generate()`, `generateStreaming()` (push callback), and `generateTokens()` (pull-based `TokenStream`) |
| `LlamaContext.TokenStream` | `Iterator<String>` + `Iterable<String>` + `AutoCloseable` view over one `generateTokens()` call; runs generation on a background thread and hands off tokens one at a time, so a caller can pull (and stop pulling) at its own pace |
| `NativeLogBridge` | Forwards llama.cpp/ggml native log output into SLF4J (see [Native log output](#native-log-output)) |
| `NativeCrashException` | Thrown when native code hits a fatal signal that was intercepted instead of killing the JVM (see [Crash containment](#crash-containment)) |

Generation uses greedy sampling when `temperature <= 0`, otherwise temperature + distribution sampling (`llama_sampler_init_temp` + `llama_sampler_init_dist`).

### Token streaming

`generateStreaming()`'s native callback runs synchronously on the calling thread - convenient for
a push-style consumer, but not directly usable as a pull-based `Iterator`/`Stream`/source for a
blocking-I/O server loop. `generateTokens()` bridges the two: it starts the (blocking) native
generation call on a dedicated background thread and hands each token to the caller through a
`SynchronousQueue`, exposed as `TokenStream`. This was chosen over `java.util.concurrent.Flow`
(Reactive Streams) because `ApiServer` is built on the JDK's thread-per-request, blocking
`HttpServer` - there's no async I/O underneath for Reactive Streams' backpressure machinery to
plug into, so a plain blocking `Iterator` matches the actual execution model with far less code.

Always close a `TokenStream` (try-with-resources, as above) even if you stop iterating before it's
exhausted: `close()` interrupts the background thread and drains the queue until it exits, so an
early `break` (e.g. a client disconnecting mid-response in `ApiServer`) can't leak it.

### Concurrency

A single `llama_context` is not reentrant: concurrent `llama_decode` calls against it corrupt its KV cache / sampler state. `LlamaContext` guards every native call with a lock, so calls on a *shared* context are serialized rather than racing — but for actual parallel generation across users, create one `LlamaContext` per concurrent request via `LlamaModel.createContext()`. The underlying `LlamaModel` is safe to share across contexts (this matches llama.cpp's own multi-slot server design). `ApiServer` follows exactly this pattern: one shared `LlamaModel` per registered model, loaded lazily on first request, and a fresh `LlamaContext` (closed when the request ends) per request - so concurrent requests against the same model run truly in parallel rather than queuing behind the lock. A context pool (to avoid paying KV-cache allocation cost on every request) would be a reasonable next optimization but isn't implemented yet.

### Crash containment

Bad input or a native bug inside llama.cpp/ggml can otherwise either escape as a C++ exception across the JNI boundary (undefined behavior) or raise a fatal signal (`SIGSEGV`, `SIGABRT` from a failed `GGML_ASSERT`, `SIGBUS`, `SIGFPE`, `SIGILL`) that kills the JVM process outright. `native/llama_jni.cpp` guards against both: arguments are validated before touching native memory, C++ exceptions are caught and rethrown as Java exceptions, and a per-call signal handler converts a caught fatal signal into a `NativeCrashException` instead of taking down the process.

This is best-effort containment, not full recovery: a signal caught mid-call may leave native heap state corrupted in ways invisible from Java, so once any call has crashed, every subsequent native call in the process refuses to run (throwing `NativeCrashException` immediately) — restart the process rather than continuing to use it. It also only covers the JNI-calling thread; a crash on one of ggml's internal worker threads (batched decode with `nThreads > 1`) is not caught.

### Native log output

llama.cpp/ggml log everything (model loading, hardware/backend detection, decode warnings, the tensor-loading progress dots) through `llama_log_set`/`ggml_log_set` rather than printing directly. `LlamaModel` registers `NativeLogBridge` before `LlamaNative.backendInit()`, which forwards every native log line into the SLF4J logger named `dev.localllm.native` (mapped to `debug`/`info`/`warn`/`error`; multi-part "continuation" output like the progress dots is coalesced into a single line first). Control verbosity the normal SLF4J/Logback way, e.g. in your own `logback.xml`:

```xml
<logger name="dev.localllm.native" level="WARN"/>
```

or override the bundled default entirely with `-Dlogback.configurationFile=/path/to/logback.xml`.
