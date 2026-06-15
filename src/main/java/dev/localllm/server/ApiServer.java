package dev.localllm.server;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import dev.localllm.model.ModelConfig;
import dev.localllm.model.ModelRegistry;
import dev.localllm.runner.ModelRunner;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * Ollama互換のREST APIサーバー。
 * GET  /api/tags      モデル一覧
 * POST /api/generate  テキスト生成
 * POST /api/chat      チャット補完
 */
public class ApiServer {

    private final int port;
    private final ModelRegistry registry;
    private final ModelRunner runner;
    private final Gson gson;

    public ApiServer(int port, ModelRegistry registry) {
        this.port = port;
        this.registry = registry;
        this.runner = new ModelRunner();
        this.gson = new GsonBuilder().setPrettyPrinting().create();
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
        String prompt    = req.get("prompt").getAsString();
        int numPredict   = 200;
        if (req.has("options")) {
            JsonObject opts = req.getAsJsonObject("options");
            if (opts.has("num_predict")) numPredict = opts.get("num_predict").getAsInt();
        }

        ModelConfig model = registry.get(modelName).orElse(null);
        if (model == null) {
            sendError(ex, 404, "model not found: " + modelName);
            return;
        }

        try {
            String generated = runner.generate(model, prompt, numPredict);
            Map<String, Object> resp = new HashMap<>();
            resp.put("model",      modelName);
            resp.put("created_at", Instant.now().toString());
            resp.put("response",   generated);
            resp.put("done",       true);
            sendJson(ex, 200, resp);
        } catch (Exception e) {
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

        String modelName  = req.get("model").getAsString();
        JsonArray messages = req.getAsJsonArray("messages");

        // ChatML形式でプロンプトを組み立てる
        StringBuilder prompt = new StringBuilder();
        for (int i = 0; i < messages.size(); i++) {
            JsonObject msg = messages.get(i).getAsJsonObject();
            prompt.append("<|im_start|>").append(msg.get("role").getAsString()).append("\n")
                  .append(msg.get("content").getAsString()).append("<|im_end|>\n");
        }
        prompt.append("<|im_start|>assistant\n");

        ModelConfig model = registry.get(modelName).orElse(null);
        if (model == null) {
            sendError(ex, 404, "model not found: " + modelName);
            return;
        }

        try {
            String generated = runner.generate(model, prompt.toString(), 500);

            Map<String, String> message = new HashMap<>();
            message.put("role",    "assistant");
            message.put("content", generated);

            Map<String, Object> resp = new HashMap<>();
            resp.put("model",      modelName);
            resp.put("created_at", Instant.now().toString());
            resp.put("message",    message);
            resp.put("done",       true);
            sendJson(ex, 200, resp);
        } catch (Exception e) {
            sendError(ex, 500, "chat failed: " + e.getMessage());
        }
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
}
