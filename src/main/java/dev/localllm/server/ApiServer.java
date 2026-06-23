package dev.localllm.server;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import dev.localllm.jni.LlamaContext;
import dev.localllm.jni.LlamaModel;
import dev.localllm.model.ModelConfig;
import dev.localllm.model.ModelRegistry;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Ollama互換のREST APIサーバー。
 * GET  /api/tags      モデル一覧
 * POST /api/generate  テキスト生成
 * POST /api/chat      チャット補完
 *
 * Generation runs in-process via the JNI binding ({@link LlamaModel} /
 * {@link LlamaContext}) instead of shelling out to a llama.cpp binary, so
 * tokens can be streamed to the HTTP response as they're produced. Each
 * registered model is loaded once on first use and kept resident; each
 * request gets its own {@link LlamaContext} (cheap relative to the model
 * itself), matching the one-context-per-concurrent-generation model
 * described on {@link LlamaContext}.
 */
public class ApiServer {

    private static final Logger LOG = LoggerFactory.getLogger(ApiServer.class);

    // Not yet exposed via ModelConfig / `add` - same defaults for every
    // model until per-model tuning is needed.
    private static final int DEFAULT_N_CTX = 4096;
    private static final int DEFAULT_N_THREADS = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final float DEFAULT_TEMPERATURE = 0.8f;
    private static final int DEFAULT_NUM_PREDICT = 200;

    private final int port;
    private final ModelRegistry registry;
    private final Map<String, LlamaModel> loadedModels = new ConcurrentHashMap<>();
    private final Gson gson;
    private final Gson ndjsonGson;

    public ApiServer(int port, ModelRegistry registry) {
        this.port = port;
        this.registry = registry;
        this.gson = new GsonBuilder().setPrettyPrinting().create();
        this.ndjsonGson = new Gson(); // one compact JSON object per line, no pretty-printing
    }

    public void start() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/api/tags", this::handleTags);
        server.createContext("/api/generate", this::handleGenerate);
        server.createContext("/api/chat", this::handleChat);
        server.createContext("/", this::handleRoot);
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();

        System.out.printf("Listening on http://localhost:%d%n", port);
        System.out.println();
        System.out.printf("  GET  http://localhost:%d/api/tags%n", port);
        System.out.printf("  POST http://localhost:%d/api/generate%n", port);
        System.out.printf("  POST http://localhost:%d/api/chat%n", port);
        System.out.println();
        System.out.println("Press Ctrl+C to stop.");

        Thread.currentThread().join();
    }

    private void handleRoot(HttpExchange ex) throws IOException {
        Map<String, String> body = new HashMap<>();
        body.put("name", "local-llm");
        body.put("version", "1.0.0");
        sendJson(ex, 200, body);
    }

    private void handleTags(HttpExchange ex) throws IOException {
        if (!"GET".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method Not Allowed");
            return;
        }

        List<ModelConfig> models = registry.list();
        List<Map<String, Object>> modelList = new ArrayList<>();
        for (ModelConfig m : models) {
            Map<String, Object> entry = new HashMap<>();
            entry.put("name", m.getName());
            entry.put("modified_at", m.getAddedAt() != null ? m.getAddedAt() : Instant.now().toString());
            entry.put("size", m.getSizeBytes());
            Map<String, String> details = new HashMap<>();
            details.put("format", m.getFormat() != null ? m.getFormat() : "gguf");
            entry.put("details", details);
            modelList.add(entry);
        }

        Map<String, Object> resp = new HashMap<>();
        resp.put("models", modelList);
        sendJson(ex, 200, resp);
    }

    private void handleGenerate(HttpExchange ex) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method Not Allowed");
            return;
        }

        String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        JsonObject req = gson.fromJson(body, JsonObject.class);

        String modelName = req.get("model").getAsString();
        String prompt = req.get("prompt").getAsString();
        GenerationOptions opts = GenerationOptions.from(req);

        ModelConfig modelConfig = registry.get(modelName).orElse(null);
        if (modelConfig == null) {
            sendError(ex, 404, "model not found: " + modelName);
            return;
        }

        LlamaModel model;
        try {
            model = getOrLoadModel(modelConfig);
        } catch (Exception e) {
            LOG.error("failed to load model '{}'", modelName, e);
            sendError(ex, 500, "failed to load model: " + e.getMessage());
            return;
        }

        try (LlamaContext ctx = model.createContext(DEFAULT_N_CTX, DEFAULT_N_THREADS);
             LlamaContext.TokenStream tokens = ctx.generateTokens(prompt, opts.numPredict, opts.temperature)) {

            if (opts.stream) {
                ex.getResponseHeaders().set("Content-Type", "application/x-ndjson");
                ex.sendResponseHeaders(200, 0); // chunked: length unknown up front
                try (OutputStream os = ex.getResponseBody()) {
                    for (String piece : tokens) {
                        writeNdjsonLine(os, generateChunk(modelName, piece, false));
                    }
                    writeNdjsonLine(os, generateChunk(modelName, "", true));
                }
            } else {
                StringBuilder sb = new StringBuilder();
                for (String piece : tokens) {
                    sb.append(piece);
                }
                sendJson(ex, 200, generateChunk(modelName, sb.toString(), true));
            }
        } catch (Exception e) {
            LOG.error("generation failed for model '{}'", modelName, e);
            sendError(ex, 500, "generation failed: " + e.getMessage());
        }
    }

    private void handleChat(HttpExchange ex) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method Not Allowed");
            return;
        }

        String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        JsonObject req = gson.fromJson(body, JsonObject.class);

        String modelName = req.get("model").getAsString();
        JsonArray messages = req.getAsJsonArray("messages");
        GenerationOptions opts = GenerationOptions.from(req);
        if (!req.has("options") || !req.getAsJsonObject("options").has("num_predict")) {
            opts.numPredict = 500; // chat replies default to a longer budget than raw /api/generate
        }

        // ChatML形式でプロンプトを組み立てる
        StringBuilder prompt = new StringBuilder();
        for (int i = 0; i < messages.size(); i++) {
            JsonObject msg = messages.get(i).getAsJsonObject();
            prompt.append("<|im_start|>").append(msg.get("role").getAsString()).append("\n")
                  .append(msg.get("content").getAsString()).append("<|im_end|>\n");
        }
        prompt.append("<|im_start|>assistant\n");

        ModelConfig modelConfig = registry.get(modelName).orElse(null);
        if (modelConfig == null) {
            sendError(ex, 404, "model not found: " + modelName);
            return;
        }

        LlamaModel model;
        try {
            model = getOrLoadModel(modelConfig);
        } catch (Exception e) {
            LOG.error("failed to load model '{}'", modelName, e);
            sendError(ex, 500, "failed to load model: " + e.getMessage());
            return;
        }

        try (LlamaContext ctx = model.createContext(DEFAULT_N_CTX, DEFAULT_N_THREADS);
             LlamaContext.TokenStream tokens = ctx.generateTokens(prompt.toString(), opts.numPredict, opts.temperature)) {

            if (opts.stream) {
                ex.getResponseHeaders().set("Content-Type", "application/x-ndjson");
                ex.sendResponseHeaders(200, 0);
                try (OutputStream os = ex.getResponseBody()) {
                    for (String piece : tokens) {
                        writeNdjsonLine(os, chatChunk(modelName, piece, false));
                    }
                    writeNdjsonLine(os, chatChunk(modelName, "", true));
                }
            } else {
                StringBuilder sb = new StringBuilder();
                for (String piece : tokens) {
                    sb.append(piece);
                }
                sendJson(ex, 200, chatChunk(modelName, sb.toString(), true));
            }
        } catch (Exception e) {
            LOG.error("chat failed for model '{}'", modelName, e);
            sendError(ex, 500, "chat failed: " + e.getMessage());
        }
    }

    private LlamaModel getOrLoadModel(ModelConfig config) {
        return loadedModels.computeIfAbsent(config.getName(), name -> new LlamaModel(config.getPath(), 0));
    }

    private static Map<String, Object> generateChunk(String modelName, String responsePiece, boolean done) {
        Map<String, Object> resp = new HashMap<>();
        resp.put("model", modelName);
        resp.put("created_at", Instant.now().toString());
        resp.put("response", responsePiece);
        resp.put("done", done);
        return resp;
    }

    private static Map<String, Object> chatChunk(String modelName, String contentPiece, boolean done) {
        Map<String, String> message = new HashMap<>();
        message.put("role", "assistant");
        message.put("content", contentPiece);

        Map<String, Object> resp = new HashMap<>();
        resp.put("model", modelName);
        resp.put("created_at", Instant.now().toString());
        resp.put("message", message);
        resp.put("done", done);
        return resp;
    }

    /** One compact JSON object per line (newline-delimited JSON), flushed immediately. */
    private void writeNdjsonLine(OutputStream os, Object obj) throws IOException {
        os.write(ndjsonGson.toJson(obj).getBytes(StandardCharsets.UTF_8));
        os.write('\n');
        os.flush();
    }

    private void sendJson(HttpExchange ex, int status, Object obj) throws IOException {
        byte[] bytes = gson.toJson(obj).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void sendError(HttpExchange ex, int status, String message) throws IOException {
        Map<String, String> err = new HashMap<>();
        err.put("error", message);
        sendJson(ex, status, err);
    }

    /** Parses the Ollama-style {@code stream} / {@code options.num_predict} / {@code options.temperature} fields. */
    private static final class GenerationOptions {
        boolean stream = true;
        int numPredict = DEFAULT_NUM_PREDICT;
        float temperature = DEFAULT_TEMPERATURE;

        static GenerationOptions from(JsonObject req) {
            GenerationOptions opts = new GenerationOptions();
            if (req.has("stream")) {
                opts.stream = req.get("stream").getAsBoolean();
            }
            if (req.has("options")) {
                JsonObject o = req.getAsJsonObject("options");
                if (o.has("num_predict")) {
                    opts.numPredict = o.get("num_predict").getAsInt();
                }
                if (o.has("temperature")) {
                    opts.temperature = o.get("temperature").getAsFloat();
                }
            }
            return opts;
        }
    }
}
