# local-llm

A lightweight Java tool for managing and running local LLMs — similar to Ollama but minimal by design.

Models are registered by pointing to a local GGUF file and a [llama.cpp](https://github.com/ggerganov/llama.cpp) binary. The registry is stored in `~/.local-llm/models.json`.

## Requirements

- Java 11+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-cli` binary) for running models
- A GGUF model file

## Build

**Without Maven** (recommended — downloads Gson automatically):

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
└── src/main/java/dev/localllm/
    ├── Main.java                         # CLI entry point
    ├── model/
    │   ├── ModelConfig.java              # Model POJO
    │   └── ModelRegistry.java            # Persists registry to ~/.local-llm/models.json
    ├── runner/
    │   └── ModelRunner.java              # Runs llama.cpp as a subprocess
    └── server/
        └── ApiServer.java                # Ollama-compatible HTTP API server
```

## Notes

- The registry file lives at `~/.local-llm/models.json` and persists across sessions.
- The API server uses the JDK's built-in `com.sun.net.httpserver.HttpServer` — no extra dependencies beyond Gson.
- Chat prompts are formatted using [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md), which is compatible with most modern GGUF models.
