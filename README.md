# local-llm / jllm

![Demo](docs/demo.gif)

A lightweight Java tool for managing and running local LLMs — similar to Ollama but minimal by design.

Models are registered by pointing to a local GGUF file. The registry is stored in `~/.local-llm/models.json`. Large GGUF files can be imported into a managed storage directory (`~/.local-llm/models/`) and removed cleanly from there with a single command.

Both `run` (interactive chat) and `serve` (HTTP API server) run inference in-process via a JNI binding to llama.cpp — no `llama-cli` subprocess required. If the JNI native library is not available, `run` falls back to a `llama-cli` subprocess automatically.

The tool is extensible: drop a JAR into `~/.local-llm/plugins/` to add new **function-calling tools** or **prompt interceptors** without rebuilding the application.

RAG (Retrieval-Augmented Generation) is built in: index local PDFs and text files with `jllm rag add`, then pass `--rag <collection>` to `jllm run` or include `"rag_collection"` in any API request. Retrieval runs entirely on-device via an embedded [Apache Lucene](https://lucene.apache.org/) index — no embedding model, no external server, no network required.

## Requirements

- Java 11+
- A GGUF model file
- For in-process inference: a built `libllamajni.so` (see [JNI Binding](#jni-binding-devlocalllmjni))
- For `run` subprocess fallback only: [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-cli` binary)

## Build

**Without Maven** (recommended — downloads Gson, SLF4J, Logback, and Undertow automatically):

```bash
bash build.sh
```

**With Maven:**

```bash
mvn package
```

Output: `target/local-llm.jar`

## The `jllm` wrapper

A thin shell wrapper is provided so you don't have to type `java -jar target/local-llm.jar` every time:

```bash
./jllm <command> [options]
```

To make it available system-wide, add the project directory to your `PATH` or copy it to `/usr/local/bin`:

```bash
cp jllm /usr/local/bin/jllm
```

All examples below use `jllm`. Substitute `java -jar target/local-llm.jar` if you prefer not to use the wrapper.

## Usage

### Commands

| Command | Description |
|---|---|
| `pull <owner>/<repo>[/<file.gguf>]` | Download a GGUF from HuggingFace and register it |
| `list` | List registered models with disk status and total size |
| `storage` | Per-model disk usage summary (managed vs. external vs. missing) |
| `add <name> --path <path>` | Register a model by pointing to a GGUF file |
| `create <name> -f <file>` | Create a model from a Modelfile or Jllmfile |
| `rm <name> [--purge]` | Remove a model from the registry (optionally delete the file) |
| `update <name> [flags]` | Modify registered model parameters in place without re-registering |
| `run <name> [flags]` | Interactive REPL **or** non-interactive one-shot (pipe-friendly) |
| `serve [--port <port>] [--max-concurrent <n>]` | Start the HTTP API server (default port: 11434) |
| `rag add <collection> <path>` | Index a file or directory into a RAG collection |
| `rag list` | List all RAG collections with chunk counts |
| `rag search <collection> <query>` | Test retrieval (shows top chunks and BM25 scores) |
| `rag rm <collection>` | Delete a RAG collection and its index |
| `verify [<name>]` | Verify SHA-256 checksum(s) against stored values |
| `show <name> [--yaml]` | Print the model's config (Modelfile or Jllmfile format) |
| `info <name>` | Show model details (includes SHA-256 and GGUF metadata) |
| `plugins` | List all loaded plugin tools and interceptors |
| `version` | Show jllm version, runtime, and dependency info |

### `pull` — Download from HuggingFace

```bash
# List available GGUF quantisations in a repo (pick one from the output)
jllm pull bartowski/Llama-3.2-3B-Instruct-GGUF

# Download a specific file — auto-registered as "Llama-3.2-3B-Instruct-Q4_K_M"
jllm pull bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Register under a shorter alias
jllm pull bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf --name llama3.2:3b

# Private / gated models — token via flag or environment variable
export HF_TOKEN=hf_...
jllm pull meta-llama/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

| Flag | Description |
|---|---|
| `--name <alias>` | Register the model under this name (default: filename without `.gguf`) |
| `--branch <ref>` | HuggingFace branch or commit SHA (default: `main`) |
| `--token <token>` | HF access token for private or gated models (or set `HF_TOKEN`) |
| `--binary <path>` | Path to `llama-cli` binary (auto-detected if omitted) |
| `--no-register` | Download only; skip registry entry |
| `--quantize <type>` | Quantize the downloaded file before registering (requires JNI; e.g. `Q4_K_M`). Produces a new `.gguf` alongside the original; the quantized file is registered |
| `--threads <int>` | CPU thread count to use during quantization (default: all cores) |

**Download behaviour:**
- Files are saved to `~/.local-llm/models/<filename>.gguf` (managed storage).
- A `.part` temp file is used during transfer and renamed on completion — a failed download leaves only the `.part` file, which is automatically cleaned up on retry.
- If the destination file already exists the download is skipped.
- Progress is shown live: `1.23 GB / 2.04 GB  [████████░░░░]  60%  18 MB/s  ETA 44s`
- **Split models** — when the downloaded file matches the `model-NNNNN-of-MMMMM.gguf` pattern, jllm automatically downloads all remaining shards before registering. In the file-listing step, only the first shard is shown (the rest are implied), with a note on how many shards will be fetched.

After pulling, run directly:
```bash
jllm run Llama-3.2-3B-Instruct-Q4_K_M   # or whatever --name you chose
```

---

### `add` — Register a model

```bash
jllm add phi3:mini --path ~/models/phi3-mini-q4.gguf --binary /usr/local/bin/llama-cli
```

| Flag | Description |
|---|---|
| `--path <path>` | **(required)** Path to the GGUF model file |
| `--binary <path>` | Path to `llama-cli`. Auto-detected if omitted (used only as subprocess fallback) |
| `--format <fmt>` | Model format (default: `gguf`) |
| `--managed` | Copy the file into `~/.local-llm/models/` (managed storage) before registering |
| `--quantize <type>` | Quantize in-process before registering (requires JNI). Supported types: `Q2_K`, `Q3_K_S/M/L`, `Q4_0`, `Q4_K_S/M`, `Q5_0`, `Q5_K_S/M`, `Q6_K`, `Q8_0`, `F16`, `BF16`. The quantized file is registered; the original is left on disk |
| `--quantize-output <path>` | Override the output path for the quantized file (default: same directory as input with the type appended to the filename stem, e.g. `phi3-f16-Q4_K_M.gguf`; with `--managed`, placed in `~/.local-llm/models/`) |
| `--threads <int>` | CPU thread count for quantization (default: all cores) |

**On-the-fly quantization** — if you have a full-precision (F16/BF16/F32) GGUF and want to compress it before registering, pass `--quantize`:

```bash
# Quantize a downloaded F16 GGUF to Q4_K_M and register the result
jllm add phi3:mini --path ~/downloads/phi3-f16.gguf --quantize Q4_K_M

# Place the quantized file in managed storage automatically
jllm add phi3:mini --path ~/downloads/phi3-f16.gguf --quantize Q4_K_M --managed

# Full pipeline: download an F16 from HuggingFace, quantize, register
jllm pull owner/repo/phi3-f16.gguf --quantize Q4_K_M --name phi3:mini
```

Progress and size reduction are printed:
```
Quantizing Q4_K_M...
  Input  : /home/user/downloads/phi3-f16.gguf
  Output : /home/user/downloads/phi3-f16-Q4_K_M.gguf
Quantization complete in 42.3 s  (7.6 GB → 2.4 GB)
```

At registration time, the GGUF binary header is parsed and a SHA-256 checksum is computed. Both are printed immediately and stored in the registry:

```
Registered 'phi3:mini' (2.2 GB)
  GGUF:    arch=phi3  quant=Q4_K_M  params=3.82B  ctx=4096
  SHA-256: a3f2d1c97e8fb5c1...
```

The GGUF metadata is visible in the `QUANT` / `PARAMS` columns of `jllm list` and in full detail via `jllm info`. The SHA-256 is used by `jllm verify` to detect file corruption or accidental replacement. If another registered model has the same SHA-256, a duplicate warning is printed:

```
  Note: same content as 'phi3:mini-backup' already registered
```

Pass `--no-hash` to skip the checksum computation (useful for very large files when you don't need integrity checking).

**Split GGUF models** — if the path points to any shard of a split model (e.g. `model-00001-of-00004.gguf`), jllm detects the naming pattern, resolves all sibling shards, registers the primary shard as the path (which llama.cpp uses to load the whole model), and records all shard paths internally:

```
$ jllm add llama70b --path ~/models/llama-00001-of-00004.gguf
  Split GGUF: 4 shards
Registered 'llama70b' (38.2 GB)
  GGUF:  arch=llama  quant=Q4_K_M  params=70.55B  ctx=4096
  SHA-256: a3f2d1c97e8fb5c1...
```

Giving a non-first shard redirects automatically to shard 1. Missing sibling files emit a warning but do not fail registration. Use `jllm info llama70b` to see the per-shard breakdown.

### `create` — Create from a config file

```bash
jllm create phi3:mini -f Modelfile
jllm create phi3:mini -f my-model.yaml
```

| Flag | Description |
|---|---|
| `-f <path>`, `--file <path>` | **(required)** Path to the Modelfile or Jllmfile |
| `--binary <path>` | Path to `llama-cli` for the subprocess fallback |

The file format is detected from the extension: `.yaml` / `.yml` / `Jllmfile` → YAML (Jllmfile); anything else → Modelfile (Ollama-compatible).

### `rm` — Remove a model

```bash
jllm rm phi3:mini              # remove from registry only; file stays on disk
jllm rm phi3:mini --purge      # remove from registry AND delete the file
```

`--purge` prints how many bytes were freed.

### `verify` — Check file integrity

```bash
jllm verify              # verify all registered models
jllm verify phi3:mini    # verify one model
```

Computes the SHA-256 of each file and compares it against the value stored at registration time. For large files (> 50 MB) a progress bar is shown during hashing.

```
Verifying 4 model(s)...

  phi3:mini                    OK           sha256:a3f2d1c97e8fb5c1
  llama70b                     OK           sha256:e9e274fffd44685a (4 shards)
  llama3.2-3b                  MISMATCH     stored:b5e6f7g8...  got:c9d0e1f2...
  old-model                    MISSING      /home/user/.local-llm/models/old.gguf
  legacy-model                 NO HASH      run: jllm update legacy-model --refresh-hash

Summary: 2 OK  1 MISMATCH  1 MISSING  1 NO HASH
```

| Status | Meaning |
|---|---|
| `OK` | File matches the stored checksum |
| `MISMATCH` | File content has changed since registration — possible corruption or accidental replacement |
| `MISSING` | Registered path no longer exists on disk |
| `NO HASH` | Model was registered before checksums were supported; run `--refresh-hash` to add one |

Exit code is `1` if any MISMATCH or MISSING is found, `0` otherwise — suitable for use in scripts or CI.

### `update` — Modify model parameters in place

```bash
# Show current config (no flags = display only)
jllm update phi3:mini

# Set or change individual parameters
jllm update phi3:mini --temperature 0.2
jllm update phi3:mini --ctx 8192 --threads 8
jllm update phi3:mini --system "You are a strict JSON generator. Output only valid JSON."
jllm update phi3:mini --max-tokens 2048

# Offload 35 layers to GPU
jllm update phi3:mini --gpu-layers 35

# Offload all layers to GPU
jllm update phi3:mini --gpu-layers -1

# Clear system prompt
jllm update phi3:mini --no-system

# Reset a parameter to the runtime default (removes it from the stored config)
jllm update phi3:mini --unset temperature
jllm update phi3:mini --unset gpu-layers   # revert to CPU-only

# Move the GGUF file — update the registered path and recalculate size
jllm update phi3:mini --path /new/location/phi3-mini.gguf
```

| Flag | Description |
|---|---|
| `--temperature <float>` | Set sampling temperature |
| `--max-tokens <int>` | Set max output tokens (`num_predict`) |
| `--ctx <int>` | Set context window size (`num_ctx`) |
| `--threads <int>` | Set CPU thread count (`num_threads`) |
| `--gpu-layers <int>` | Set number of model layers to offload to GPU (`num_gpu_layers`). `0` = CPU only, `-1` = all layers, positive integer = partial offload |
| `--system <text>` | Set system prompt |
| `--no-system` | Clear the stored system prompt |
| `--path <path>` | Update GGUF file path (also recalculates size and re-reads GGUF metadata) |
| `--binary <path>` | Update the `llama-cli` binary path |
| `--refresh-gguf` | Re-read GGUF metadata from the current file (useful for models added before this feature existed) |
| `--unset <param>` | Reset a parameter to `null` (runtime default). Params: `temperature` \| `max-tokens` \| `ctx` \| `threads` \| `gpu-layers` \| `system` |

Only the flags you pass are changed; everything else stays the same. Changes are
shown as a before→after diff:

```
Updated 'phi3:mini':
  temperature    (not set)  →  0.2
  num_ctx        (not set)  →  8192
  system         (not set)  →  "You are a strict JSON generator..."
```

### `show` — Print model config

```bash
jllm show phi3:mini            # Modelfile (Ollama-compatible) format
jllm show phi3:mini --yaml     # Jllmfile (YAML) format
```

### `serve` — Start the HTTP API server

```bash
jllm serve
jllm serve --port 8080
jllm serve --max-concurrent 4
```

| Flag | Description |
|---|---|
| `--port <port>` | Port to listen on (default: `11434`) |
| `--max-concurrent <n>` | Maximum simultaneous inference slots (default: CPU core count). Also controls the number of concurrent sequences in the continuous-batch scheduler. |

On **Java 21+**, each HTTP request is handled on a **Virtual Thread** (Project Loom) — created instantly, with no OS thread per connection. On **Java 11–20**, a cached platform thread pool is used instead.

**Continuous batching (JNI mode):** When the JNI library is available, inference is handled by a `BatchScheduler` per model. Incoming requests are queued and decoded together in a single `llama_decode` call per step — up to `--max-concurrent` sequences at once. This amortises GPU/CPU kernel launch overhead across concurrent users and is the same approach used by llama.cpp's own server. Each sequence gets its own KV-cache slot and sampler chain so outputs are independent. Scheduler metrics (active sequences, pending requests) are exposed via `GET /api/ps`.

**Fallback (no JNI):** If the native library is not available, inference falls back to the `ContextPool` path: `LlamaContext` instances are pooled and reused across requests, one request active per context at a time. The pool holds up to `--max-concurrent` idle contexts per model configuration.

### `version` — Show environment info

```bash
jllm version
```

```
jllm 0.1.0

Runtime
  Java    : 11.0.25 (Ubuntu)
  JVM     : OpenJDK 64-Bit Server VM 11.0.25+9
  OS      : Linux 5.15.0-89-generic (aarch64)

JNI
  Status  : available

Dependencies
  Gson             2.10.1
  SLF4J            2.0.13
  Logback          1.5.6
  Undertow         2.3.14.Final
  XNIO             3.8.14.Final
  Apache Lucene    9.11.1
  Apache PDFBox    3.0.3

Storage
  Registry:  /home/user/.local-llm/models.json
  Models:    /home/user/.local-llm/models
  Plugins:   /home/user/.local-llm/plugins
  RAG:       /home/user/.local-llm/rag
```

Useful for bug reports and verifying the active JNI library status.

---

## RAG — local document search (`jllm rag`)

RAG lets the model answer questions about your own files without sending anything outside the machine.  
Documents are indexed once into a named **collection**; at chat time jllm retrieves the most relevant passages and injects them into the model's context automatically.

Supported file types: **PDF** (text extracted page-by-page via PDFBox), plus any plain-text format (`.txt`, `.md`, `.java`, `.py`, `.json`, `.yaml`, `.html`, `.csv`, `.sql`, …).

### How it works

1. **Index** — jllm reads each file, splits it into ~400-word chunks with 50-word overlap, and stores them in a [Lucene](https://lucene.apache.org/) BM25 full-text index at `~/.local-llm/rag/<collection>/`.
2. **Retrieve** — at the start of each chat turn (or each API request), jllm queries the index with the user's message and retrieves the top-5 most relevant chunks.
3. **Generate** — the retrieved chunks are prepended to the model's system prompt as a `[Context from local documents]` block. The model uses them to answer and cites the source file (and page number for PDFs).

No embedding model is needed — BM25 is fast, accurate for keyword-rich queries, and requires zero configuration.

### `jllm rag add` — Index documents

```bash
# Index a single PDF
jllm rag add my-docs ~/papers/attention-is-all-you-need.pdf

# Index an entire directory (recursively; unsupported files are skipped)
jllm rag add my-docs ~/documents/

# Build multiple collections for different topics
jllm rag add api-specs    ~/work/specs/
jllm rag add legal-docs   ~/contracts/
```

Re-indexing a file that was already indexed is safe — the old chunks are replaced automatically, so `rag add` is idempotent.

### `jllm rag list` — List collections

```bash
jllm rag list
```

```
COLLECTION                  CHUNKS  PATH
---------------------------------------------------------------------------
api-specs                      248  /home/user/.local-llm/rag/api-specs
legal-docs                      91  /home/user/.local-llm/rag/legal-docs
my-docs                         57  /home/user/.local-llm/rag/my-docs
```

### `jllm rag search` — Test retrieval

Debug which chunks would be injected for a given query, without running the model:

```bash
jllm rag search my-docs "transformer attention mechanism"
```

```
[1] attention-is-all-you-need.pdf (page 3)  score=4.821
    Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention"...

[2] attention-is-all-you-need.pdf (page 4)  score=3.104
    Multi-Head Attention Instead of performing a single attention function with d_model-dimensional...
```

### `jllm rag rm` — Delete a collection

```bash
jllm rag rm my-docs
# → Deleted collection 'my-docs'.
```

### Using RAG in an interactive session

Pass `--rag <collection>` to `jllm run`:

```bash
jllm run phi3:mini --rag my-docs
```

```
Model    : phi3:mini
Settings : temperature=0.80  max_tokens=512  context=4096
RAG      : collection 'my-docs' (top-5 chunks per turn)
Commands : /clear  /save [file]  /help  /quit
------------------------------------------------------------

You> What does the paper say about multi-head attention?

Assistant> According to the paper (page 4), multi-head attention runs h attention
functions in parallel on projected versions of the queries, keys, and values.
Each "head" focuses on different positional subspaces, and the results are
concatenated and projected back to the full dimension...
```

RAG context is retrieved fresh on every turn so the model always uses the most relevant passages for each question. `/clear` resets the conversation history but does not affect the index.

### Using RAG via the HTTP API

Include `"rag_collection"` in any request body. The last `user`-role message is used as the retrieval query:

```bash
# Ollama-compatible endpoint
curl http://localhost:11434/api/chat \
  -d '{
    "model": "phi3:mini",
    "messages": [{"role":"user","content":"What is the conclusion of the report?"}],
    "rag_collection": "my-docs"
  }'

# OpenAI-compatible endpoint
curl http://localhost:11434/v1/chat/completions \
  -d '{
    "model": "phi3:mini",
    "messages": [{"role":"user","content":"Summarise section 3."}],
    "rag_collection": "legal-docs",
    "stream": true
  }'

# Plain text generation (/api/generate, /v1/completions) — prompt used as query
curl http://localhost:11434/api/generate \
  -d '{
    "model": "phi3:mini",
    "prompt": "Explain the attention formula from the paper.",
    "rag_collection": "my-docs"
  }'
```

Each API request can reference a different collection; collections are shared across concurrent requests.

### Storage layout (RAG)

```
~/.local-llm/rag/
├── my-docs/          # Lucene index — one directory per collection
│   ├── segments_N
│   ├── _0.cfe
│   ├── _0.cfs
│   └── write.lock
└── legal-docs/
    └── ...
```

Lucene index files are managed entirely by jllm. Do not edit them by hand.

---

## Interactive chat and scripting (`jllm run`)

`jllm run` operates in two modes selected automatically:

| Situation | Mode |
|---|---|
| Terminal (no pipe) | **Interactive REPL** — streaming chat, multi-turn history |
| `--prompt <text>` given | **Non-interactive** — generate once, print to stdout, exit |
| stdin is piped (`\|` or `<`) | **Non-interactive** — read stdin as prompt, generate, exit |

### Interactive REPL

```bash
jllm run phi3:mini
```

Loads the model in-process via JNI and opens a streaming terminal REPL. Tokens appear character-by-character as they are generated.

```
Model    : phi3:mini
Settings : temperature=0.80  max_tokens=512  context=4096
Commands : /clear  /save [file]  /help  /quit
------------------------------------------------------------

You> Tell me a joke.

Assistant> Why don't scientists trust atoms?
Because they make up everything.

You> /clear
[History cleared]

You> /quit
Goodbye!
```

**REPL commands:**

| Command | Description |
|---|---|
| `/clear` | Clear conversation history and reset context |
| `/save [file]` | Save the conversation log to a Markdown file (auto-named if omitted) |
| `/help` | Show available commands and loaded tools |
| `/quit` | Exit (also: `/exit`, `/bye`, Ctrl+D) |

**Multi-turn context:** conversation history is maintained across turns using ChatML format. Each generation re-processes the full accumulated history from position 0.

### Non-interactive / pipe mode

When stdin is not a terminal, `jllm run` automatically switches to one-shot mode: it reads the full prompt, generates a response, streams it to stdout, and exits. No banner, no `You>` / `Assistant>` prefixes — stdout is the raw model output, ready for piping.

```bash
# Inline prompt via flag
jllm run phi3:mini --prompt "Explain BM25 in one sentence."

# Pipe from echo
echo "What is the capital of Japan?" | jllm run phi3:mini

# Pipe file content
cat report.txt | jllm run phi3:mini --system "Summarise the following text."

# Capture output into a variable
ANSWER=$(jllm run phi3:mini --prompt "What is 2 + 2?")

# Batch processing from a file
while IFS= read -r q; do
    echo "Q: $q"
    echo "A: $(jllm run phi3:mini --prompt "$q" --temperature 0)"
done < questions.txt

# RAG + pipe — document Q&A in a script
echo "What is the conclusion?" | jllm run phi3:mini --rag my-docs
```

llama-cli native logs go to stderr and can be silenced with `2>/dev/null` without affecting the answer in stdout.

### Session flags

All flags below apply to both interactive and non-interactive mode. They override the values stored in the model's Modelfile/Jllmfile **for this session only** — the registry entry is never modified.

| Flag | Description |
|---|---|
| `--prompt <text>` | Prompt text for non-interactive one-shot generation |
| `--rag <collection>` | Enable RAG retrieval from the named collection |
| `--system <prompt>` | Replace the model's system prompt |
| `--no-system` | Clear the system prompt entirely |
| `--temperature <float>` | Sampling temperature (e.g. `0.0` for greedy, `1.2` for creative) |
| `--max-tokens <int>` | Maximum tokens to generate (alias: `--num-predict`) |
| `--ctx <int>` | Context window size in tokens (alias: `--num-ctx`) |
| `--threads <int>` | CPU thread count for inference (alias: `--num-threads`) |
| `--gpu-layers <int>` | GPU layers to offload for this session (`0` = CPU only, `-1` = all; alias: `--num-gpu-layers`) |

Examples:

```bash
# Act as a translator for this session only
jllm run phi3:mini --system "Translate every message to English. Output only the translation."

# Deterministic output for testing (temperature 0 = greedy)
jllm run phi3:mini --temperature 0 --prompt "What is 7 × 8?"

# Larger context for a long document session
jllm run phi3:mini --ctx 16384 --rag legal-docs

# Erase the model's built-in persona and go bare
jllm run phi3:mini --no-system --temperature 0.9
```

**Subprocess fallback:** if `libllamajni.so` is not available, `run` falls back to launching `llama-cli` as a subprocess (requires `--binary` at registration time). Both interactive and non-interactive modes support the fallback.

---

## Modelfile (Ollama-compatible format)

`create` reads a **Modelfile** — a plain-text file that Ollama also understands.
Every field is optional except `FROM`.

```
# My assistant
FROM /path/to/model.gguf

PARAMETER temperature    0.7
PARAMETER num_predict    1024
PARAMETER num_ctx        4096
PARAMETER num_threads    4
PARAMETER num_gpu_layers 35

SYSTEM You are a helpful, concise assistant.
```

Multi-line system prompts use triple-quote blocks:

```
SYSTEM """
You are a helpful assistant.
Always respond in the language the user writes in.
"""
```

Supported `PARAMETER` keys:

| Key | Type | Description |
|---|---|---|
| `temperature` | float | Sampling temperature |
| `num_predict` | int | Maximum tokens to generate |
| `num_ctx` | int | Context window size in tokens |
| `num_threads` | int | CPU thread count |
| `num_gpu_layers` | int | Layers to offload to GPU (`0` = CPU only, `-1` = all, positive = partial offload) |

Unknown instructions (e.g. `TEMPLATE`, `ADAPTER`) are silently ignored, so Modelfiles
written for full Ollama can be reused here without parse errors.

---

## Jllmfile (YAML format)

As an alternative to Modelfile, `create` also accepts a **Jllmfile** — a YAML file.
Use `.yaml`, `.yml`, or the literal filename `Jllmfile` as the extension.

```yaml
# Jllmfile
from: /path/to/model.gguf
binary: /usr/local/bin/llama-cli

system: "You are a helpful, concise assistant."

parameters:
  temperature: 0.7
  num_predict: 1024
  num_ctx: 4096
  num_threads: 4
  num_gpu_layers: 35
```

Multi-line system prompts use YAML block scalars:

```yaml
system: |
  You are a helpful assistant.
  Always respond in the language the user writes in.
```

Supported `parameters` keys:

| Key | Type | Description |
|---|---|---|
| `temperature` | float | Sampling temperature |
| `num_predict` | int | Maximum tokens to generate |
| `num_ctx` | int | Context window size in tokens |
| `num_threads` | int | CPU thread count |
| `num_gpu_layers` | int | Layers to offload to GPU (`0` = CPU only, `-1` = all, positive = partial offload) |

Unknown keys are silently ignored for forward compatibility.

---

## Disk storage management

### `jllm list` — disk status at a glance

```
NAME                      QUANT      PARAMS   SHARDS  SIZE       STATUS    PATH
-----------------------------------------------------------------------------------------------------------
phi3:mini                 Q4_K_M     3.82B    -       2.4 GB     ok        /home/user/.local-llm/models/phi3.gguf
llama70b                  Q4_K_M     70.55B   4       38.2 GB    ok        /home/user/.local-llm/models/llama-00001-of-00004.gguf
old-model                 -          -        -       4.1 GB     missing   /mnt/external/old.gguf
-----------------------------------------------------------------------------------------------------------
3 model(s)  40.6 GB total on disk  (1 file(s) missing — run 'storage' for details)
```

The `SHARDS` column shows `-` for single-file models or the shard count for split models.


`STATUS` can be:
- `ok` — file exists on disk
- `missing` — registered path no longer exists (stale entry)

### `jllm storage` — full disk usage view

```bash
jllm storage
```

```
Managed storage dir: /home/user/.local-llm/models

NAME                      SIZE       STATUS     PATH
------------------------------------------------------------------------------------------
phi3:mini                 2.4 GB     managed    /home/user/.local-llm/models/phi3.gguf
llama3:8b                 4.9 GB     external   /downloads/llama3.gguf
old-model                 4.1 GB     missing    /mnt/external/old.gguf
------------------------------------------------------------------------------------------
Total: 3 model(s)  7.3 GB on disk
       1 file(s) missing — remove stale entries with: jllm rm <name>

Status legend:
  managed   file lives under the managed storage dir (safe to purge via jllm rm --purge)
  external  file is registered by path but not copied to managed storage
  missing   registered path no longer exists on disk
```

### Managed storage workflow

Import an existing GGUF file into managed storage (copies it to `~/.local-llm/models/`):

```bash
jllm add phi3:mini --path ~/downloads/phi3-mini-q4.gguf --managed
```

Once the model is managed, you can safely delete it and free the disk space with a single command:

```bash
jllm rm phi3:mini --purge
# → Removed 'phi3:mini' from registry.
# → Deleted file: /home/user/.local-llm/models/phi3-mini-q4.gguf (2.4 GB freed)
```

---

## Plugin architecture

jllm supports drop-in JAR plugins that add new capabilities without rebuilding the application.
Plugins are discovered at startup from `~/.local-llm/plugins/`.

Two plugin types are available:

| Type | Interface | Effect |
|---|---|---|
| **Tool** | `LlmTool` | Function-calling tool the model can invoke during a conversation |
| **Interceptor** | `PromptInterceptor` | Transforms the assembled ChatML prompt before each generation call |

### Listing loaded plugins

```bash
jllm plugins
```

```
Plugin directory: /home/user/.local-llm/plugins

Tools (1):
  NAME                  DESCRIPTION                                    SOURCE JAR
  --------------------------------------------------------------------------------
  weather               Get current weather for a location             weather-1.0.jar

Interceptors (1):
  PRIORITY  CLASS                                     SOURCE JAR
  ----------------------------------------------------------------------
  100       LoggingInterceptor                        logging-plugin.jar
```

### How tool calling works

When at least one tool is loaded, jllm automatically appends tool-use instructions to the system prompt before each generation call. The model is told to reply with:

```
<tool_call>{"name":"TOOL_NAME","args":{...}}</tool_call>
```

jllm detects that pattern in the response, executes the tool, injects the result back into the conversation as a user message, and continues generation — up to 5 tool calls per user turn. This mechanism is model-agnostic and works with any instruction-following model.

```
You> What's the weather in Tokyo?

Assistant> <tool_call>{"name":"weather","args":{"location":"Tokyo"}}</tool_call>
[Tool result] Sunny, 28°C

Assistant> The current weather in Tokyo is sunny with a temperature of 28°C.
```

### How interceptors work

`PromptInterceptor.intercept(prompt)` receives the fully assembled ChatML prompt string and returns a (possibly modified) version. Interceptors are sorted by `getPriority()` (ascending) and applied as a pipeline. They run in both `jllm run` and `jllm serve`.

Use cases: prompt logging, keyword substitution, adding a preamble or context injection.

### Building a plugin JAR

**Step 1 — Implement the interface:**

```java
// EchoTool.java
import dev.localllm.plugin.LlmTool;
import com.google.gson.*;

public class EchoTool implements LlmTool {
    @Override public String getName()             { return "echo"; }
    @Override public String getDescription()      { return "Echoes the provided text back verbatim."; }
    @Override public String getParametersSchema() {
        return "{\"type\":\"object\","
             + "\"properties\":{\"text\":{\"type\":\"string\"}},"
             + "\"required\":[\"text\"]}";
    }
    @Override public String execute(String argsJson) throws Exception {
        return new Gson().fromJson(argsJson, JsonObject.class)
                         .get("text").getAsString();
    }
}
```

**Step 2 — Register via Java SPI:**

Create `META-INF/services/dev.localllm.plugin.LlmTool` (or `...PromptInterceptor`) containing the fully-qualified class name:

```
EchoTool
```

**Step 3 — Build and install:**

```bash
# Compile against the fat JAR (which contains the plugin interfaces)
javac -cp /path/to/local-llm.jar EchoTool.java

# Package
jar cf echo-tool.jar EchoTool.class META-INF/

# Install
mkdir -p ~/.local-llm/plugins/
cp echo-tool.jar ~/.local-llm/plugins/

# Verify
jllm plugins
```

Full working examples (with build scripts) are in `examples/plugins/`:

| Directory | Plugin type | Description |
|---|---|---|
| `echo-tool/` | `LlmTool` | Echoes back the text argument |
| `logging-interceptor/` | `PromptInterceptor` | Appends every prompt to `~/.local-llm/prompt.log` |

### Thread safety

Plugin instances are shared across all concurrent requests in `jllm serve` mode. Both `LlmTool.execute()` and `PromptInterceptor.intercept()` may be called from multiple threads simultaneously — implementations must be thread-safe.

---

## Parameter precedence

At inference time, parameters are resolved in this order:

1. `options.*` in the API request body (highest — caller always wins)
2. `PARAMETER` / `parameters:` values from the model's Modelfile or Jllmfile
3. CLI session flags (e.g. `--gpu-layers`, `--temperature`) — override registry values for one invocation only
4. Server-wide defaults (`temperature 0.8`, `num_predict 200`, `num_ctx 4096`, `num_gpu_layers 0`)

`num_gpu_layers` is applied at model-load time, so it takes effect when the model is first loaded into memory. In `jllm serve`, the model is cached after the first request — changing `num_gpu_layers` in the registry requires restarting the server to take effect.

---

## Examples

```bash
# Download from HuggingFace and register automatically
jllm pull bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf --name llama3.2:3b

# List available quantisations first, then pick one
jllm pull bartowski/Phi-3.5-mini-instruct-GGUF

# Register a model by path (no config file)
jllm add phi3:mini --path ~/models/phi3-mini-q4.gguf --binary /usr/local/bin/llama-cli

# Register and import into managed storage
jllm add phi3:mini --path ~/models/phi3-mini-q4.gguf --managed

# Create from a Modelfile (Ollama-compatible)
jllm create phi3:mini -f Modelfile

# Create from a Jllmfile (YAML)
jllm create phi3:mini -f phi3.yaml

# List all models (with disk status and total size)
jllm list

# Show detailed disk usage
jllm storage

# Show model config as Modelfile
jllm show phi3:mini

# Show model config as Jllmfile (YAML)
jllm show phi3:mini --yaml

# Show raw model details
jllm info phi3:mini

# Interactive streaming chat
jllm run phi3:mini

# Interactive chat with RAG over a document collection
jllm rag add my-docs ~/documents/
jllm run phi3:mini --rag my-docs

# One-shot prompt (non-interactive, output goes to stdout)
jllm run phi3:mini --prompt "Explain BM25 in one sentence."

# Pipe a file through the model
cat report.txt | jllm run phi3:mini --system "Summarise the following text."

# Capture output into a shell variable
ANSWER=$(jllm run phi3:mini --prompt "Capital of Japan?" --max-tokens 20)

# Session overrides — temperature, context window, system prompt
jllm run phi3:mini --temperature 0 --ctx 8192 --system "You are a strict JSON generator."

# Offload all layers to GPU for this session only
jllm run phi3:mini --gpu-layers -1

# Persist GPU offload in the registry
jllm update phi3:mini --gpu-layers 35

# On-the-fly quantization: register a Q4_K_M version of an F16 GGUF
jllm add phi3:mini --path ~/downloads/phi3-f16.gguf --quantize Q4_K_M --managed

# Download F16 from HuggingFace, quantize to Q4_K_M in-process, register
jllm pull owner/repo/phi3-f16.gguf --quantize Q4_K_M --name phi3:mini

# Start API server on the default port (11434)
jllm serve

# Start API server on a custom port
jllm serve --port 8080

# Limit to 2 simultaneous inference calls (excess requests queue)
jllm serve --max-concurrent 2

# Remove from registry (file stays on disk)
jllm rm phi3:mini

# Remove from registry AND delete the file
jllm rm phi3:mini --purge

# List loaded plugins
jllm plugins
```

---

## HTTP API

The embedded HTTP server is built on **Undertow** and exposes both an Ollama-compatible API
(`/api/...`) and an OpenAI-compatible API (`/v1/...`).

### Concurrency model

Each incoming request is dispatched to a dedicated thread:

- **Java 21+** — a **Virtual Thread** (Project Loom). Virtual threads are created instantly (no OS thread per connection), so thousands of concurrent SSE clients or chat sessions cost almost nothing in memory or scheduling overhead.
- **Java 11–20** — a platform thread from a cached thread pool.

**Inference dispatch** depends on whether the JNI library is available:

| Path | When | Mechanism |
|---|---|---|
| **BatchScheduler** (primary) | JNI library available | Continuous batching — all active sequences decoded in one `llama_decode` call per step |
| **ContextPool** (fallback) | No JNI library | One request per pooled context; semaphore limits concurrency |

With the `BatchScheduler`, a request's HTTP handler submits tokenised input to the scheduler and blocks on a `BlockingQueue`, reading tokens as they arrive. The scheduler thread runs independently, mixing prefill and decode steps across all queued sequences. Up to `--max-concurrent` sequences are active simultaneously; additional requests wait in a bounded queue.

Streaming behaviour:
- **Ollama endpoints** stream newline-delimited JSON (`application/x-ndjson`), one object per token.
- **OpenAI endpoints** stream Server-Sent Events (`text/event-stream`), one `data:` line per token,
  terminated with `data: [DONE]`.

Pass `"stream": false` in the request body to receive a single JSON object instead of a stream.

All endpoints respond with `Access-Control-Allow-Origin: *` CORS headers so browser clients can
connect directly. `OPTIONS` preflight requests are handled automatically.

The model named in `model` is loaded into memory on first use and kept resident for subsequent
requests (no reload-per-request).

**Prompt interceptors** loaded from `~/.local-llm/plugins/` are applied to every prompt in all
handlers — both Ollama and OpenAI endpoints.

### Endpoints

| Method | Path | Protocol | Description |
|---|---|---|---|
| `GET`  | `/api/tags`             | Ollama  | List registered models |
| `POST` | `/api/show`             | Ollama  | Model details (Modelfile + parameters) |
| `POST` | `/api/generate`         | Ollama  | Text generation |
| `POST` | `/api/chat`             | Ollama  | Chat completion |
| `POST` | `/api/embeddings`       | Ollama  | Text embeddings |
| `GET`  | `/api/ps`               | Ollama  | Context pool stats and idle context count |
| `GET`  | `/v1/models`            | OpenAI  | List models |
| `POST` | `/v1/chat/completions`  | OpenAI  | Chat completion |
| `POST` | `/v1/completions`       | OpenAI  | Text completion |
| `POST` | `/v1/embeddings`        | OpenAI  | Text embeddings |

### OpenAI: `GET /v1/models`

```bash
curl http://localhost:11434/v1/models
```

```json
{
  "object": "list",
  "data": [
    { "id": "phi3:mini", "object": "model", "created": 1719600000, "owned_by": "local-llm" }
  ]
}
```

### OpenAI: `POST /v1/chat/completions`

```bash
curl http://localhost:11434/v1/chat/completions \
  -d '{
    "model": "phi3:mini",
    "messages": [{ "role": "user", "content": "Hello!" }],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": true
  }'
```

Streaming SSE response (default):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1719600000,"model":"phi3:mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1719600000,"model":"phi3:mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1719600000,"model":"phi3:mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Non-streaming (`"stream": false`):

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1719600000,
  "model": "phi3:mini",
  "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Hello! How can I help?" }, "finish_reason": "stop" }],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

### OpenAI: `POST /v1/completions`

```bash
curl http://localhost:11434/v1/completions \
  -d '{ "model": "phi3:mini", "prompt": "Once upon a time", "max_tokens": 128 }'
```

### Embeddings: `POST /api/embeddings` and `POST /v1/embeddings`

Requires the JNI native library. The model runs a dedicated embedding context (no KV cache is shared with inference requests) and returns an L2-normalized float vector.

**Ollama format** (`/api/embeddings`):

```bash
curl http://localhost:11434/api/embeddings \
  -d '{ "model": "nomic-embed-text", "prompt": "Hello, world!" }'
```

```json
{
  "model": "nomic-embed-text",
  "embedding": [0.023, -0.011, 0.047, "..."]
}
```

**OpenAI format** (`/v1/embeddings`):

```bash
curl http://localhost:11434/v1/embeddings \
  -d '{ "model": "nomic-embed-text", "input": "Hello, world!" }'
```

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "embedding": [0.023, -0.011, 0.047, "..."], "index": 0 }
  ],
  "model": "nomic-embed-text",
  "usage": { "prompt_tokens": 0, "total_tokens": 0 }
}
```

Both endpoints also accept `"input"` as an array — only the first element is embedded (single-vector response). The vector dimension matches the model's `n_embd`. BERT-style encoder models (`nomic-embed-text`, `mxbai-embed-large`, etc.) and decoder-based embedding models (LLaMA with `--embeddings`) are both supported.

### Ollama: `POST /api/show` — Model details

```bash
curl http://localhost:11434/api/show -d '{"name": "phi3:mini"}'
```

```json
{
  "modelfile": "FROM ~/models/phi3-mini.gguf\nPARAMETER temperature 0.7\nSYSTEM You are a helpful assistant.\n",
  "parameters": "temperature 0.7\nnum_predict 1024",
  "details": { "format": "gguf" }
}
```

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

### `GET /api/ps` — Inference status

```bash
curl http://localhost:11434/api/ps
```

```json
{
  "models": [],
  "batch_schedulers": [
    {
      "name": "phi3:mini",
      "active_sequences": 2,
      "pending_requests": 1,
      "num_ctx": "4096",
      "num_threads": "4"
    }
  ],
  "pool_stats": {
    "hits": 0,
    "misses": 0,
    "hit_rate": "0.0%",
    "total_idle": 0,
    "evictions": 0
  }
}
```

In JNI mode:
- **`batch_schedulers`** — one entry per active `(model, num_ctx, num_threads)` tuple. `active_sequences` is the number of requests currently being decoded; `pending_requests` is the number queued waiting for a slot.
- **`models`** — empty (ContextPool is not used in JNI mode).

In fallback (no JNI) mode:
- **`models`** — one entry per idle context; `idle_contexts` is the number of pooled contexts ready to reuse.
- **`batch_schedulers`** — empty.
- **`pool_stats`** — `hit_rate` is the fraction of requests that reused a pooled context.

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

---

## Project Structure

```
local-llm-env/
├── build.sh                              # Maven-free build script
├── jllm                                  # Shell wrapper (runs target/local-llm.jar)
├── pom.xml                               # Maven build file
├── examples/
│   └── plugins/
│       ├── echo-tool/                    # Example LlmTool plugin (with build.sh)
│       └── logging-interceptor/          # Example PromptInterceptor plugin
├── native/                               # JNI wrapper around llama.cpp's C API
│   ├── CMakeLists.txt
│   ├── build.sh
│   └── llama_jni.cpp
├── native/dist/                          # Pre-built native libs for JAR bundling
│   ├── linux-x86_64/libllamajni.so
│   ├── linux-x86_64-cuda/libllamajni.so
│   ├── osx-aarch64/libllamajni.dylib
│   └── windows-x86_64/llamajni.dll
├── src/main/resources/
│   └── logback.xml                       # Default Logback config
└── src/main/java/dev/localllm/
    ├── Main.java                         # CLI entry point (all sub-commands)
    ├── Version.java                      # Compile-time version constants for all bundled deps
    ├── model/
    │   ├── ModelConfig.java              # Model POJO (path, shards, parameters, system prompt, GGUF metadata, sha256)
    │   ├── GgufReader.java               # GGUF binary header parser (KV section only; no tensor data)
    │   ├── SplitGguf.java                # Split GGUF detection (3 naming patterns) and shard enumeration
    │   ├── Modelfile.java                # Modelfile parser and serializer (Ollama-compatible)
    │   ├── JllmfileParser.java           # Jllmfile parser and serializer (YAML format)
    │   └── ModelRegistry.java            # Persists registry to ~/.local-llm/models.json
    ├── plugin/
    │   ├── LlmTool.java                  # SPI: function-calling tool interface
    │   ├── PromptInterceptor.java        # SPI: prompt transformation interface
    │   └── PluginManager.java            # JAR scanner, URLClassLoader, interceptor chain
    ├── pull/
    │   └── HuggingFaceClient.java        # HF Hub API: list GGUF files + stream download
    ├── rag/
    │   ├── RagResult.java                # Search result POJO (source, page, chunk, BM25 score)
    │   ├── DocumentChunker.java          # Split text into overlapping word-level chunks
    │   ├── DocumentReader.java           # Read PDF (PDFBox) and plain-text files
    │   ├── RagIndex.java                 # Lucene FSDirectory wrapper (search + stats)
    │   └── RagManager.java               # Collection management, indexing, context block builder
    ├── runner/
    │   └── ModelRunner.java              # Interactive REPL: JNI streaming + tool calling loop + RAG
    ├── server/
    │   ├── ApiServer.java                # Undertow HTTP server: Ollama + OpenAI APIs + RAG
    │   ├── BatchScheduler.java           # Continuous-batch scheduler: multi-seq llama_decode loop
    │   └── ContextPool.java              # LlamaContext pool: KV cache reuse (JNI fallback)
    └── jni/
        ├── LlamaNative.java              # Raw native method declarations (single + batch primitives, quantize)
        ├── QuantizeType.java             # Enum mapping quantization type names to llama_ftype integer values
        ├── NativeLibraryLoader.java      # Locates and loads libllamajni.so
        ├── NativeLogBridge.java          # Forwards native log output into SLF4J
        ├── NativeCrashException.java     # Thrown when a native fatal signal is caught
        ├── LlamaModel.java               # High-level model wrapper (AutoCloseable); createContext + createBatchContext
        ├── BatchContext.java             # Multi-sequence context wrapper used by BatchScheduler
        ├── LlamaContext.java             # Single-sequence inference context wrapper (ContextPool fallback)
        └── LlamaDemo.java                # Minimal smoke-test CLI
```

### Storage layout

```
~/.local-llm/
├── models.json          # Registry: all registered model metadata
├── models/              # Managed storage (populated by jllm add --managed)
│   ├── phi3-mini.gguf
│   └── llama3-8b.gguf
├── plugins/             # Plugin JARs (drop-in; loaded at startup)
│   ├── weather-1.0.jar
│   └── logging-plugin.jar
└── rag/                 # RAG Lucene indices (one subdirectory per collection)
    ├── my-docs/         # jllm rag add my-docs ~/documents/
    └── api-specs/       # jllm rag add api-specs ~/work/specs/
```

---

## Notes

- The registry file `~/.local-llm/models.json` persists across sessions. All config parameters (`temperature`, `num_predict`, `num_ctx`, `num_threads`, `num_gpu_layers`, system prompt) are stored as part of each model's entry. A `null` value for any parameter means "use the runtime default" — for `num_gpu_layers` the default is `0` (CPU only).
- The API server is built on [Undertow](https://undertow.io/) (embedded, no servlet container needed). It adds ~3 MB to the fat JAR.
- On Java 21+, the server uses **Virtual Threads** (Project Loom) for lightweight multi-user concurrency. The source is compiled with `-source 11` for compatibility; virtual thread support is detected and enabled at runtime via reflection. No code changes or flags are needed — run with Java 21+ and virtual threads activate automatically.
- Chat prompts are formatted using [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md). A model's `SYSTEM` prompt is injected as a `system` turn at the start of every chat — unless the request already includes a `system` role message, in which case the request takes precedence.
- Logging goes through SLF4J ([Logback](https://logback.qos.ch/) by default, see `src/main/resources/logback.xml`); this includes llama.cpp/ggml's own native log output (see [Native log output](#native-log-output) below). In interactive REPL mode, native INFO logs are suppressed after the model and context load — they only appear at startup.
- Plugin JARs are each loaded in an isolated `URLClassLoader` (child of the application classloader), so multiple plugins with conflicting class names coexist safely.
- RAG retrieval uses [Apache Lucene](https://lucene.apache.org/) BM25 (the same ranking algorithm used by Elasticsearch and OpenSearch). No embedding model or vector database is required. PDF text extraction is handled by [Apache PDFBox](https://pdfbox.apache.org/). Both libraries are bundled in the fat JAR — no extra setup needed.
- The RAG index is persistent: you index once and reuse across many sessions. Re-indexing a file replaces its previous chunks so the operation is always safe to repeat.

---

## JNI Binding (`dev.localllm.jni`)

In addition to the subprocess-based fallback, this project includes a direct **JNI binding to llama.cpp's C API**, so Java code can run inference in-process (no `llama-cli` subprocess, no stdout parsing). This is the default path for both `jllm run` and `jllm serve`.

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

    // Streaming via pull-based Iterator/Iterable (used by ApiServer):
    try (LlamaContext.TokenStream tokens = ctx.generateTokens("Once upon a time", 128, 0.8f)) {
        for (String piece : tokens) {
            System.out.print(piece);
        }
    }
}
```

### Smoke test

```bash
java -Ddev.localllm.nativeLibDir=native/build -cp target/classes \
  dev.localllm.jni.LlamaDemo /path/to/model.gguf "Once upon a time"
```

Verified end-to-end against [`ggml-org/models/tinyllamas/stories260K.gguf`](https://huggingface.co/ggml-org/models) (a 1.2 MB test model), producing coherent greedy-sampled output.

### API surface

| Class | Purpose |
|---|---|
| `LlamaNative` | Raw `native` method declarations mirroring llama.cpp's C API — single-sequence `generate`, multi-sequence batch primitives (`newBatchContext`, `batchDecode`, `samplerCreate/Sample/Free`, `kvSeqRm`), on-the-fly `quantize`, plus model/context lifecycle, tokenization, log callback |
| `LlamaModel` | `AutoCloseable` wrapper; loads a GGUF file, exposes tokenization helpers, `createContext()`, and `createBatchContext()` |
| `BatchContext` | `AutoCloseable` multi-sequence context; wraps the six batch JNI primitives used by `BatchScheduler` |
| `LlamaContext` | `AutoCloseable` single-sequence context; exposes `generate()`, `generateStreaming()` (push callback), and `generateTokens()` (pull-based `TokenStream`) — used by the ContextPool fallback |
| `LlamaContext.TokenStream` | `Iterator<String>` + `Iterable<String>` + `AutoCloseable` view over one `generateTokens()` call; runs generation on a background thread and hands off tokens one at a time |
| `NativeLogBridge` | Forwards llama.cpp/ggml native log output into SLF4J (see [Native log output](#native-log-output)) |
| `NativeCrashException` | Thrown when native code hits a fatal signal that was intercepted instead of killing the JVM (see [Crash containment](#crash-containment)) |

Generation uses greedy sampling when `temperature <= 0`, otherwise temperature + distribution sampling (`llama_sampler_init_temp` + `llama_sampler_init_dist`).

### Token streaming

`generateStreaming()`'s native callback runs synchronously on the calling thread — convenient for
a push-style consumer, but not directly usable as a pull-based `Iterator`/`Stream`/source for a
blocking-I/O server loop. `generateTokens()` bridges the two: it starts the (blocking) native
generation call on a dedicated background thread and hands each token to the caller through a
`SynchronousQueue`, exposed as `TokenStream`. This was chosen over `java.util.concurrent.Flow`
(Reactive Streams) because `ApiServer` is built on the JDK's thread-per-request, blocking
`HttpServer` — there's no async I/O underneath for Reactive Streams' backpressure machinery to
plug into, so a plain blocking `Iterator` matches the actual execution model with far less code.

Always close a `TokenStream` (try-with-resources, as above) even if you stop iterating before it's
exhausted: `close()` interrupts the background thread and drains the queue until it exits, so an
early `break` (e.g. a client disconnecting mid-response in `ApiServer`) can't leak it.

### Concurrency

A single `llama_context` is not reentrant: concurrent `llama_decode` calls against it corrupt its KV cache / sampler state. The underlying `LlamaModel` is safe to share across contexts (this matches llama.cpp's own multi-slot server design).

`ApiServer` creates one shared `LlamaModel` per registered model (loaded lazily on first request). Concurrent requests are handled by the **`BatchScheduler`** — a dedicated background thread that drives a multi-sequence `BatchContext`. Each sequence gets its own `seq_id` (0 … maxSeqs−1) in the shared KV cache; llama.cpp filters attention by `seq_id`, so independent sequences do not interfere even when they occupy the same position indices. Up to `--max-concurrent` sequences are decoded together in each `llama_decode` call.

When the JNI library is unavailable, `ApiServer` falls back to the **`ContextPool`**: multiple `LlamaContext` instances are pooled and reused across requests, one request active per context at a time. Pool metrics are exposed at `GET /api/ps`.

### Crash containment

Bad input or a native bug inside llama.cpp/ggml can otherwise either escape as a C++ exception across the JNI boundary (undefined behavior) or raise a fatal signal (`SIGSEGV`, `SIGABRT` from a failed `GGML_ASSERT`, `SIGBUS`, `SIGFPE`, `SIGILL`) that kills the JVM process outright. `native/llama_jni.cpp` guards against both: arguments are validated before touching native memory, C++ exceptions are caught and rethrown as Java exceptions, and a per-call signal handler converts a caught fatal signal into a `NativeCrashException` instead of taking down the process.

This is best-effort containment, not full recovery: a signal caught mid-call may leave native heap state corrupted in ways invisible from Java, so once any call has crashed, every subsequent native call in the process refuses to run (throwing `NativeCrashException` immediately) — restart the process rather than continuing to use it. It also only covers the JNI-calling thread; a crash on one of ggml's internal worker threads (batched decode with `nThreads > 1`) is not caught.

### Native log output

llama.cpp/ggml log everything (model loading, hardware/backend detection, decode warnings, the tensor-loading progress dots) through `llama_log_set`/`ggml_log_set` rather than printing directly. `LlamaModel` registers `NativeLogBridge` before `LlamaNative.backendInit()`, which forwards every native log line into the SLF4J logger named `dev.localllm.native` (mapped to `debug`/`info`/`warn`/`error`; multi-part "continuation" output like the progress dots is coalesced into a single line first). Control verbosity the normal SLF4J/Logback way, e.g. in your own `logback.xml`:

```xml
<logger name="dev.localllm.native" level="WARN"/>
```

or override the bundled default entirely with `-Dlogback.configurationFile=/path/to/logback.xml`.
