package dev.localllm.server;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import dev.localllm.jni.LlamaContext;
import dev.localllm.jni.LlamaModel;
import dev.localllm.model.ModelConfig;
import dev.localllm.model.Modelfile;
import dev.localllm.model.ModelRegistry;
import dev.localllm.plugin.PluginManager;
import io.undertow.Handlers;
import io.undertow.Undertow;
import io.undertow.server.HttpHandler;
import io.undertow.server.HttpServerExchange;
import io.undertow.server.handlers.BlockingHandler;
import io.undertow.util.Headers;
import io.undertow.util.HttpString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Embedded HTTP server with both Ollama-compatible and OpenAI-compatible REST APIs.
 *
 * Ollama API:
 *   GET  /api/tags               list registered models
 *   POST /api/show               model details (Modelfile + parameters)
 *   POST /api/generate           text generation  — NDJSON stream
 *   POST /api/chat               chat completion  — NDJSON stream
 *
 * OpenAI API:
 *   GET  /v1/models              list models
 *   POST /v1/chat/completions    chat completion  — SSE stream
 *   POST /v1/completions         text completion  — SSE stream
 *
 * Streaming behaviour:
 *   Ollama endpoints use newline-delimited JSON (application/x-ndjson).
 *   OpenAI endpoints use Server-Sent Events (text/event-stream), terminated
 *   with the conventional "data: [DONE]" line.
 *   Pass "stream": false in the request body to receive a single JSON object.
 *
 * Inference runs in-process via the JNI binding (LlamaModel / LlamaContext).
 * Each registered model is loaded once on first use and kept resident;
 * each HTTP request gets its own short-lived LlamaContext.
 */
public class ApiServer {

    private static final Logger LOG = LoggerFactory.getLogger(ApiServer.class);

    // Server-wide defaults — overridden by Modelfile PARAMETER values, which
    // are in turn overridden by per-request options.
    private static final int   DEFAULT_N_CTX      = 4096;
    private static final int   DEFAULT_N_THREADS  = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final float DEFAULT_TEMPERATURE = 0.8f;
    private static final int   DEFAULT_NUM_PREDICT = 200;
    private static final int   DEFAULT_CHAT_PREDICT = 500;

    // Pre-allocated HttpString instances for headers used on every response.
    private static final HttpString HDR_CORS_ORIGIN  = new HttpString("Access-Control-Allow-Origin");
    private static final HttpString HDR_CORS_METHODS = new HttpString("Access-Control-Allow-Methods");
    private static final HttpString HDR_CORS_HEADERS = new HttpString("Access-Control-Allow-Headers");
    private static final HttpString HDR_CACHE_CTRL   = new HttpString("Cache-Control");
    private static final HttpString HDR_X_ACCEL_BUF  = new HttpString("X-Accel-Buffering");

    private final int port;
    private final ModelRegistry registry;
    private final PluginManager plugins;
    private final Map<String, LlamaModel> loadedModels = new ConcurrentHashMap<>();
    private final Gson prettyGson;
    private final Gson compactGson;

    public ApiServer(int port, ModelRegistry registry) {
        this(port, registry, PluginManager.EMPTY);
    }

    public ApiServer(int port, ModelRegistry registry, PluginManager plugins) {
        this.port        = port;
        this.registry    = registry;
        this.plugins     = plugins != null ? plugins : PluginManager.EMPTY;
        this.prettyGson  = new GsonBuilder().setPrettyPrinting().create();
        this.compactGson = new Gson();
    }

    public void start() throws Exception {
        HttpHandler router = Handlers.routing()
            // root info
            .get("/",                     b(this::handleRoot))
            // ── Ollama API ────────────────────────────────────────────────────
            .get("/api/tags",             b(this::handleTags))
            .post("/api/show",            b(this::handleShow))
            .post("/api/generate",        b(this::handleGenerate))
            .post("/api/chat",            b(this::handleChat))
            // ── OpenAI API ───────────────────────────────────────────────────
            .get("/v1/models",            b(this::handleV1Models))
            .post("/v1/chat/completions", b(this::handleV1ChatCompletions))
            .post("/v1/completions",      b(this::handleV1Completions));

        Undertow server = Undertow.builder()
            .addHttpListener(port, "0.0.0.0")
            .setHandler(withCors(router))
            .build();
        server.start();

        System.out.printf("Listening on http://localhost:%d%n", port);
        System.out.println();
        System.out.println("Ollama-compatible:");
        System.out.printf("  GET  /api/tags              http://localhost:%d/api/tags%n", port);
        System.out.printf("  POST /api/show              http://localhost:%d/api/show%n", port);
        System.out.printf("  POST /api/generate          http://localhost:%d/api/generate%n", port);
        System.out.printf("  POST /api/chat              http://localhost:%d/api/chat%n", port);
        System.out.println();
        System.out.println("OpenAI-compatible:");
        System.out.printf("  GET  /v1/models             http://localhost:%d/v1/models%n", port);
        System.out.printf("  POST /v1/chat/completions   http://localhost:%d/v1/chat/completions%n", port);
        System.out.printf("  POST /v1/completions        http://localhost:%d/v1/completions%n", port);
        System.out.println();
        System.out.println("Press Ctrl+C to stop.");

        Thread.currentThread().join();
    }

    // ── Undertow helpers ──────────────────────────────────────────────────────

    /** Wrap a handler so it runs on the blocking worker thread pool. */
    private static HttpHandler b(HttpHandler h) {
        return new BlockingHandler(h);
    }

    /**
     * Add CORS headers to every response and handle OPTIONS preflight on the
     * IO thread (no blocking I/O needed for a 204).
     */
    private static HttpHandler withCors(HttpHandler next) {
        return exchange -> {
            exchange.getResponseHeaders()
                .put(HDR_CORS_ORIGIN,  "*")
                .put(HDR_CORS_METHODS, "GET, POST, OPTIONS")
                .put(HDR_CORS_HEADERS, "Content-Type, Authorization");
            if ("OPTIONS".equals(exchange.getRequestMethod().toString())) {
                exchange.setStatusCode(204);
                exchange.endExchange();
                return;
            }
            next.handleRequest(exchange);
        };
    }

    // ── Root ──────────────────────────────────────────────────────────────────

    private void handleRoot(HttpServerExchange ex) throws Exception {
        sendJson(ex, 200, map("name", "local-llm", "version", "1.0.0"));
    }

    // ── Ollama: GET /api/tags ─────────────────────────────────────────────────

    private void handleTags(HttpServerExchange ex) throws Exception {
        List<ModelConfig> models = registry.list();
        List<Map<String, Object>> list = new ArrayList<>();
        for (ModelConfig m : models) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("name",        m.getName());
            entry.put("modified_at", m.getAddedAt() != null ? m.getAddedAt() : Instant.now().toString());
            entry.put("size",        m.getSizeBytes());
            entry.put("details",     map("format", m.getFormat() != null ? m.getFormat() : "gguf"));
            list.add(entry);
        }
        sendJson(ex, 200, map("models", list));
    }

    // ── Ollama: POST /api/show ────────────────────────────────────────────────

    private void handleShow(HttpServerExchange ex) throws Exception {
        JsonObject req  = parseJson(ex);
        String name     = req.get("name").getAsString();
        ModelConfig m   = registry.get(name).orElse(null);
        if (m == null) { sendError(ex, 404, "model not found: " + name); return; }

        StringBuilder params = new StringBuilder();
        if (m.getTemperature() != null) params.append("temperature ").append(m.getTemperature()).append('\n');
        if (m.getNumPredict()  != null) params.append("num_predict " ).append(m.getNumPredict() ).append('\n');
        if (m.getNumCtx()      != null) params.append("num_ctx "     ).append(m.getNumCtx()     ).append('\n');
        if (m.getNumThreads()  != null) params.append("num_threads " ).append(m.getNumThreads() ).append('\n');

        sendJson(ex, 200, map(
            "modelfile",  Modelfile.toText(m),
            "parameters", params.toString().stripTrailing(),
            "details",    map("format", m.getFormat() != null ? m.getFormat() : "gguf")
        ));
    }

    // ── Ollama: POST /api/generate ────────────────────────────────────────────

    private void handleGenerate(HttpServerExchange ex) throws Exception {
        JsonObject req    = parseJson(ex);
        String modelName  = req.get("model").getAsString();
        String prompt     = req.get("prompt").getAsString();
        GenOpts opts      = GenOpts.fromOllama(req);

        ModelConfig cfg = requireModel(ex, modelName); if (cfg == null) return;
        LlamaModel model = loadModel(ex, cfg);         if (model == null) return;

        opts.applyModelDefaults(cfg);
        String effectivePrompt = plugins.applyInterceptors(withSystemPrompt(cfg.getSystemPrompt(), prompt));
        int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
        int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;

        try (LlamaContext ctx    = model.createContext(nCtx, nThreads);
             LlamaContext.TokenStream ts = ctx.generateTokens(effectivePrompt, opts.numPredict, opts.temperature)) {

            if (opts.stream) {
                beginNdjson(ex);
                try (OutputStream os = ex.getOutputStream()) {
                    for (String piece : ts) writeNdjson(os, ollamaGenerateChunk(modelName, piece, false));
                    writeNdjson(os, ollamaGenerateChunk(modelName, "", true));
                }
            } else {
                StringBuilder sb = new StringBuilder();
                for (String t : ts) sb.append(t);
                sendJson(ex, 200, ollamaGenerateChunk(modelName, sb.toString(), true));
            }
        } catch (Exception e) {
            LOG.error("generate failed for '{}'", modelName, e);
            sendError(ex, 500, "generation failed: " + e.getMessage());
        }
    }

    // ── Ollama: POST /api/chat ────────────────────────────────────────────────

    private void handleChat(HttpServerExchange ex) throws Exception {
        JsonObject req    = parseJson(ex);
        String modelName  = req.get("model").getAsString();
        JsonArray messages = req.getAsJsonArray("messages");
        GenOpts opts      = GenOpts.fromOllama(req);
        if (!GenOpts.ollamaHasNumPredict(req)) opts.numPredict = DEFAULT_CHAT_PREDICT;

        ModelConfig cfg = requireModel(ex, modelName); if (cfg == null) return;
        LlamaModel model = loadModel(ex, cfg);         if (model == null) return;

        opts.applyModelDefaults(cfg);
        String prompt = plugins.applyInterceptors(chatMlPrompt(messages, cfg.getSystemPrompt()));
        int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
        int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;

        try (LlamaContext ctx    = model.createContext(nCtx, nThreads);
             LlamaContext.TokenStream ts = ctx.generateTokens(prompt, opts.numPredict, opts.temperature)) {

            if (opts.stream) {
                beginNdjson(ex);
                try (OutputStream os = ex.getOutputStream()) {
                    for (String piece : ts) writeNdjson(os, ollamaChatChunk(modelName, piece, false));
                    writeNdjson(os, ollamaChatChunk(modelName, "", true));
                }
            } else {
                StringBuilder sb = new StringBuilder();
                for (String t : ts) sb.append(t);
                sendJson(ex, 200, ollamaChatChunk(modelName, sb.toString(), true));
            }
        } catch (Exception e) {
            LOG.error("chat failed for '{}'", modelName, e);
            sendError(ex, 500, "chat failed: " + e.getMessage());
        }
    }

    // ── OpenAI: GET /v1/models ────────────────────────────────────────────────

    private void handleV1Models(HttpServerExchange ex) throws Exception {
        List<Map<String, Object>> data = new ArrayList<>();
        for (ModelConfig m : registry.list()) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("id",       m.getName());
            entry.put("object",   "model");
            entry.put("created",  epochOf(m.getAddedAt()));
            entry.put("owned_by", "local-llm");
            data.add(entry);
        }
        sendJson(ex, 200, map("object", "list", "data", data));
    }

    // ── OpenAI: POST /v1/chat/completions ─────────────────────────────────────

    private void handleV1ChatCompletions(HttpServerExchange ex) throws Exception {
        JsonObject req    = parseJson(ex);
        String modelName  = req.get("model").getAsString();
        JsonArray messages = req.getAsJsonArray("messages");
        GenOpts opts      = GenOpts.fromOpenAi(req);

        ModelConfig cfg = requireModel(ex, modelName); if (cfg == null) return;
        LlamaModel model = loadModel(ex, cfg);         if (model == null) return;

        opts.applyModelDefaults(cfg);
        String prompt = plugins.applyInterceptors(chatMlPrompt(messages, cfg.getSystemPrompt()));
        int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
        int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;
        String id    = "chatcmpl-" + shortUuid();
        long created = Instant.now().getEpochSecond();

        try (LlamaContext ctx    = model.createContext(nCtx, nThreads);
             LlamaContext.TokenStream ts = ctx.generateTokens(prompt, opts.numPredict, opts.temperature)) {

            if (opts.stream) {
                beginSse(ex);
                try (OutputStream os = ex.getOutputStream()) {
                    // First chunk carries the role header
                    writeSse(os, openAiChatChunk(id, modelName, created, "", "assistant", null));
                    for (String piece : ts) writeSse(os, openAiChatChunk(id, modelName, created, piece, null, null));
                    writeSse(os, openAiChatChunk(id, modelName, created, "", null, "stop"));
                    os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                    os.flush();
                }
            } else {
                StringBuilder sb = new StringBuilder();
                for (String t : ts) sb.append(t);
                sendJson(ex, 200, openAiChatResponse(id, modelName, created, sb.toString()));
            }
        } catch (Exception e) {
            LOG.error("v1/chat/completions failed for '{}'", modelName, e);
            sendError(ex, 500, "generation failed: " + e.getMessage());
        }
    }

    // ── OpenAI: POST /v1/completions ──────────────────────────────────────────

    private void handleV1Completions(HttpServerExchange ex) throws Exception {
        JsonObject req   = parseJson(ex);
        String modelName = req.get("model").getAsString();
        String prompt    = req.get("prompt").getAsString();
        GenOpts opts     = GenOpts.fromOpenAi(req);

        ModelConfig cfg = requireModel(ex, modelName); if (cfg == null) return;
        LlamaModel model = loadModel(ex, cfg);         if (model == null) return;

        opts.applyModelDefaults(cfg);
        String effectivePrompt = plugins.applyInterceptors(withSystemPrompt(cfg.getSystemPrompt(), prompt));
        int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
        int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;
        String id    = "cmpl-" + shortUuid();
        long created = Instant.now().getEpochSecond();

        try (LlamaContext ctx    = model.createContext(nCtx, nThreads);
             LlamaContext.TokenStream ts = ctx.generateTokens(effectivePrompt, opts.numPredict, opts.temperature)) {

            if (opts.stream) {
                beginSse(ex);
                try (OutputStream os = ex.getOutputStream()) {
                    for (String piece : ts) writeSse(os, openAiCompletionChunk(id, modelName, created, piece, null));
                    writeSse(os, openAiCompletionChunk(id, modelName, created, "", "stop"));
                    os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                    os.flush();
                }
            } else {
                StringBuilder sb = new StringBuilder();
                for (String t : ts) sb.append(t);
                sendJson(ex, 200, openAiCompletionResponse(id, modelName, created, sb.toString()));
            }
        } catch (Exception e) {
            LOG.error("v1/completions failed for '{}'", modelName, e);
            sendError(ex, 500, "generation failed: " + e.getMessage());
        }
    }

    // ── HTTP I/O helpers ──────────────────────────────────────────────────────

    private JsonObject parseJson(HttpServerExchange ex) throws Exception {
        String body = new String(ex.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        return prettyGson.fromJson(body, JsonObject.class);
    }

    private void sendJson(HttpServerExchange ex, int status, Object obj) throws Exception {
        byte[] bytes = prettyGson.toJson(obj).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().put(Headers.CONTENT_TYPE, "application/json");
        ex.setStatusCode(status);
        ex.setResponseContentLength(bytes.length);
        ex.getOutputStream().write(bytes);
    }

    private void sendError(HttpServerExchange ex, int status, String message) throws Exception {
        sendJson(ex, status, map("error", message));
    }

    private void beginNdjson(HttpServerExchange ex) {
        ex.getResponseHeaders().put(Headers.CONTENT_TYPE, "application/x-ndjson");
        ex.setStatusCode(200);
    }

    private void writeNdjson(OutputStream os, Object obj) throws Exception {
        os.write(compactGson.toJson(obj).getBytes(StandardCharsets.UTF_8));
        os.write('\n');
        os.flush();
    }

    private void beginSse(HttpServerExchange ex) {
        ex.getResponseHeaders()
            .put(Headers.CONTENT_TYPE,  "text/event-stream")
            .put(HDR_CACHE_CTRL,  "no-cache")
            .put(HDR_X_ACCEL_BUF, "no");
        ex.setStatusCode(200);
    }

    private void writeSse(OutputStream os, Object obj) throws Exception {
        String line = "data: " + compactGson.toJson(obj) + "\n\n";
        os.write(line.getBytes(StandardCharsets.UTF_8));
        os.flush();
    }

    // ── Model helpers ─────────────────────────────────────────────────────────

    private ModelConfig requireModel(HttpServerExchange ex, String name) throws Exception {
        ModelConfig cfg = registry.get(name).orElse(null);
        if (cfg == null) { sendError(ex, 404, "model not found: " + name); return null; }
        return cfg;
    }

    private LlamaModel loadModel(HttpServerExchange ex, ModelConfig cfg) throws Exception {
        try {
            return loadedModels.computeIfAbsent(cfg.getName(), n -> new LlamaModel(cfg.getPath(), 0));
        } catch (Exception e) {
            LOG.error("failed to load model '{}'", cfg.getName(), e);
            sendError(ex, 500, "failed to load model: " + e.getMessage());
            return null;
        }
    }

    // ── Prompt helpers ────────────────────────────────────────────────────────

    /**
     * Build a ChatML prompt from an array of role/content messages.
     * If no system message is present in the array and the model has a
     * system prompt configured, it is prepended automatically.
     */
    private static String chatMlPrompt(JsonArray messages, String systemPrompt) {
        boolean hasSystem = false;
        for (int i = 0; i < messages.size(); i++) {
            if ("system".equals(messages.get(i).getAsJsonObject().get("role").getAsString())) {
                hasSystem = true;
                break;
            }
        }
        StringBuilder sb = new StringBuilder();
        if (!hasSystem && systemPrompt != null && !systemPrompt.isEmpty()) {
            sb.append("<|im_start|>system\n").append(systemPrompt).append("<|im_end|>\n");
        }
        for (int i = 0; i < messages.size(); i++) {
            JsonObject msg = messages.get(i).getAsJsonObject();
            sb.append("<|im_start|>").append(msg.get("role").getAsString()).append("\n")
              .append(msg.get("content").getAsString()).append("<|im_end|>\n");
        }
        return sb.append("<|im_start|>assistant\n").toString();
    }

    /** Wrap a raw prompt with a system preamble in ChatML format. */
    private static String withSystemPrompt(String system, String prompt) {
        if (system == null || system.isEmpty()) return prompt;
        return "<|im_start|>system\n" + system + "<|im_end|>\n"
             + "<|im_start|>user\n"   + prompt + "<|im_end|>\n"
             + "<|im_start|>assistant\n";
    }

    // ── Response builders ─────────────────────────────────────────────────────

    private static Map<String, Object> ollamaGenerateChunk(String model, String response, boolean done) {
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("model",      model);
        r.put("created_at", Instant.now().toString());
        r.put("response",   response);
        r.put("done",       done);
        return r;
    }

    private static Map<String, Object> ollamaChatChunk(String model, String content, boolean done) {
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("model",      model);
        r.put("created_at", Instant.now().toString());
        r.put("message",    map("role", "assistant", "content", content));
        r.put("done",       done);
        return r;
    }

    /**
     * One chunk in the OpenAI chat/completions SSE stream.
     * role: non-null only for the very first chunk.
     * finishReason: non-null only for the final (empty) chunk.
     */
    private static Map<String, Object> openAiChatChunk(
            String id, String model, long created, String content, String role, String finishReason) {
        Map<String, Object> delta = new LinkedHashMap<>();
        if (role != null)          delta.put("role",    role);
        if (role != null || !content.isEmpty()) delta.put("content", content);

        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index",         0);
        choice.put("delta",         delta);
        choice.put("finish_reason", finishReason);

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("id",      id);
        r.put("object",  "chat.completion.chunk");
        r.put("created", created);
        r.put("model",   model);
        r.put("choices", Collections.singletonList(choice));
        return r;
    }

    private static Map<String, Object> openAiChatResponse(
            String id, String model, long created, String content) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index",         0);
        choice.put("message",       map("role", "assistant", "content", content));
        choice.put("finish_reason", "stop");

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("id",      id);
        r.put("object",  "chat.completion");
        r.put("created", created);
        r.put("model",   model);
        r.put("choices", Collections.singletonList(choice));
        r.put("usage",   map("prompt_tokens", 0, "completion_tokens", 0, "total_tokens", 0));
        return r;
    }

    private static Map<String, Object> openAiCompletionChunk(
            String id, String model, long created, String text, String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index",         0);
        choice.put("text",          text);
        choice.put("finish_reason", finishReason);

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("id",      id);
        r.put("object",  "text_completion");
        r.put("created", created);
        r.put("model",   model);
        r.put("choices", Collections.singletonList(choice));
        return r;
    }

    private static Map<String, Object> openAiCompletionResponse(
            String id, String model, long created, String text) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index",         0);
        choice.put("text",          text);
        choice.put("finish_reason", "stop");

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("id",      id);
        r.put("object",  "text_completion");
        r.put("created", created);
        r.put("model",   model);
        r.put("choices", Collections.singletonList(choice));
        r.put("usage",   map("prompt_tokens", 0, "completion_tokens", 0, "total_tokens", 0));
        return r;
    }

    // ── Misc helpers ──────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private static <K, V> Map<K, V> map(Object... pairs) {
        Map<Object, Object> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < pairs.length; i += 2) m.put(pairs[i], pairs[i + 1]);
        return (Map<K, V>) m;
    }

    private static long epochOf(String isoInstant) {
        if (isoInstant == null) return Instant.now().getEpochSecond();
        try { return Instant.parse(isoInstant).getEpochSecond(); }
        catch (Exception e) { return Instant.now().getEpochSecond(); }
    }

    private static String shortUuid() {
        return UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    // ── Generation options ────────────────────────────────────────────────────

    /**
     * Parsed generation parameters from either an Ollama or OpenAI request.
     * Modelfile PARAMETER values fill in un-set fields; explicit request values
     * always win.
     */
    private static final class GenOpts {
        boolean stream      = true;
        int     numPredict  = DEFAULT_NUM_PREDICT;
        float   temperature = DEFAULT_TEMPERATURE;
        boolean numPredictSet  = false;
        boolean temperatureSet = false;

        /** Parse Ollama-style request: options.num_predict / options.temperature */
        static GenOpts fromOllama(JsonObject req) {
            GenOpts o = new GenOpts();
            if (req.has("stream")) o.stream = req.get("stream").getAsBoolean();
            if (req.has("options")) {
                JsonObject opts = req.getAsJsonObject("options");
                if (opts.has("num_predict"))  { o.numPredict  = opts.get("num_predict").getAsInt();   o.numPredictSet  = true; }
                if (opts.has("temperature"))  { o.temperature = opts.get("temperature").getAsFloat(); o.temperatureSet = true; }
            }
            return o;
        }

        /** Parse OpenAI-style request: max_tokens / temperature at root level */
        static GenOpts fromOpenAi(JsonObject req) {
            GenOpts o = new GenOpts();
            if (req.has("stream"))      o.stream = req.get("stream").getAsBoolean();
            if (req.has("max_tokens"))  { o.numPredict  = req.get("max_tokens").getAsInt();   o.numPredictSet  = true; }
            if (req.has("temperature")) { o.temperature = req.get("temperature").getAsFloat(); o.temperatureSet = true; }
            return o;
        }

        static boolean ollamaHasNumPredict(JsonObject req) {
            return req.has("options") && req.getAsJsonObject("options").has("num_predict");
        }

        void applyModelDefaults(ModelConfig m) {
            if (!numPredictSet  && m.getNumPredict()  != null) numPredict  = m.getNumPredict();
            if (!temperatureSet && m.getTemperature() != null) temperature = m.getTemperature();
        }
    }
}
