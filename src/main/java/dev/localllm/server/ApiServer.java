package dev.localllm.server;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import dev.localllm.jni.LlamaContext;
import dev.localllm.jni.LlamaModel;
import dev.localllm.server.BatchScheduler;
import dev.localllm.model.GgufReader;
import dev.localllm.model.ModelConfig;
import dev.localllm.model.Modelfile;
import dev.localllm.model.ModelRegistry;
import dev.localllm.model.SplitGguf;
import dev.localllm.pull.HuggingFaceClient;
import dev.localllm.plugin.LlmTool;
import dev.localllm.plugin.PluginManager;
import dev.localllm.plugin.PromptInterceptor;
import dev.localllm.rag.RagManager;
import dev.localllm.rag.RagResult;
import io.undertow.Handlers;
import io.undertow.Undertow;
import io.undertow.server.HttpHandler;
import io.undertow.server.HttpServerExchange;
import io.undertow.util.Headers;
import io.undertow.util.HttpString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardWatchEventKinds;
import java.nio.file.WatchKey;
import java.nio.file.WatchService;
import java.time.Instant;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Embedded HTTP server with both Ollama-compatible and OpenAI-compatible REST APIs.
 *
 * <h2>Thread model</h2>
 * On Java 21+, each HTTP request is dispatched to a new <b>Virtual Thread</b>
 * (Project Loom). Virtual threads are JVM-managed, extremely lightweight (a few
 * hundred bytes each), and unmount from their carrier OS thread whenever they block
 * at the Java level — so thousands of concurrent connections need only a handful of
 * OS threads. On older JVMs, a cached platform-thread pool is used instead.
 *
 * <p>Inference (JNI-level {@code llama_decode}) runs on the {@link LlamaContext.TokenStream}
 * producer thread, not the handler's virtual thread. The handler blocks on
 * {@link java.util.concurrent.SynchronousQueue#take()} between tokens — a pure Java
 * block — so the carrier OS thread is released between every token during SSE streaming.
 *
 * <h2>Concurrency control</h2>
 * A {@link Semaphore} limits the number of simultaneous LLM inference calls.
 * Because LLM inference is CPU-intensive (each call may saturate all cores), allowing
 * unbounded concurrent inferences would thrash the machine. The limit is set with
 * {@code --max-concurrent} (default: number of CPU cores) and only counts active
 * context allocations; requests waiting for a slot are parked cheaply on virtual threads.
 *
 * <h2>APIs</h2>
 * Ollama:  GET /api/tags  POST /api/show  POST /api/generate  POST /api/chat<br>
 * OpenAI:  GET /v1/models  POST /v1/chat/completions  POST /v1/completions
 *
 * <p>Pass {@code "stream": false} for a single JSON response instead of a stream.
 * All endpoints include {@code Access-Control-Allow-Origin: *} CORS headers.
 */
public class ApiServer {

    private static final Logger LOG = LoggerFactory.getLogger(ApiServer.class);

    // Server-wide defaults — overridden by Modelfile PARAMETER values, which
    // are in turn overridden by per-request options.
    private static final int   DEFAULT_N_CTX        = 4096;
    private static final int   DEFAULT_N_THREADS    = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final float DEFAULT_TEMPERATURE   = 0.8f;
    private static final int   DEFAULT_NUM_PREDICT   = 200;
    private static final int   DEFAULT_CHAT_PREDICT  = 500;
    private static final int   DEFAULT_MAX_CONCURRENT = Runtime.getRuntime().availableProcessors();

    // Safety defaults (overridden via ServerConfig / CLI flags)
    static final int DEFAULT_MAX_BODY_BYTES   = 4 * 1024 * 1024; // 4 MB
    static final int DEFAULT_MAX_OUTPUT_TOKENS = 0;               // 0 = no cap
    static final int DEFAULT_RATE_LIMIT_PER_MIN = 0;              // 0 = disabled

    // Matches the <tool_call>{...}</tool_call> tag the model emits for function calling.
    private static final Pattern TOOL_CALL_RE =
        Pattern.compile("<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", Pattern.DOTALL);

    // Pre-allocated HttpString instances for headers used on every response.
    private static final HttpString HDR_CORS_ORIGIN  = new HttpString("Access-Control-Allow-Origin");
    private static final HttpString HDR_CORS_METHODS = new HttpString("Access-Control-Allow-Methods");
    private static final HttpString HDR_CORS_HEADERS = new HttpString("Access-Control-Allow-Headers");
    private static final HttpString HDR_CACHE_CTRL   = new HttpString("Cache-Control");
    private static final HttpString HDR_X_ACCEL_BUF  = new HttpString("X-Accel-Buffering");

    // Detected once at class-load time; true on Java 21+.
    private static final boolean VIRTUAL_THREADS_AVAILABLE = detectVirtualThreads();

    private final int port;
    private final ModelRegistry registry;
    private final PluginManager plugins;
    private final RagManager ragManager;
    private final int maxConcurrent;
    private final int maxBodyBytes;
    private final int maxOutputTokens;
    private final IpRateLimiter rateLimiter; // null = disabled

    // Each HTTP request runs on its own virtual thread (Java 21+) or daemon
    // platform thread (Java < 21). Switching from BlockingHandler's default
    // XNIO worker pool to this executor is the only change needed for Loom.
    private final ExecutorService requestExecutor;

    // Caps simultaneous LLM inference calls. Requests that exceed the limit
    // park cheaply on virtual threads instead of creating more OS threads.
    private final Semaphore inferenceSemaphore;

    // Reuses LlamaContext instances (i.e. KV cache buffers) across requests to
    // avoid repeated native-heap allocation. Size = maxConcurrent so the pool
    // can absorb every context that could ever be in flight simultaneously.
    private final ContextPool contextPool;

    private final Map<String, LlamaModel> loadedModels = new ConcurrentHashMap<>();
    // One BatchScheduler per unique (model, nCtx, nThreads) configuration.
    private final Map<String, BatchScheduler> batchSchedulers = new ConcurrentHashMap<>();
    private final MetricsCollector metrics = new MetricsCollector();
    private final Gson prettyGson;
    private final Gson compactGson;

    public ApiServer(int port, ModelRegistry registry) {
        this(port, registry, PluginManager.EMPTY, null, DEFAULT_MAX_CONCURRENT, new ServerConfig());
    }

    public ApiServer(int port, ModelRegistry registry, PluginManager plugins) {
        this(port, registry, plugins, null, DEFAULT_MAX_CONCURRENT, new ServerConfig());
    }

    public ApiServer(int port, ModelRegistry registry, PluginManager plugins, int maxConcurrent) {
        this(port, registry, plugins, null, maxConcurrent, new ServerConfig());
    }

    public ApiServer(int port, ModelRegistry registry, PluginManager plugins,
                     RagManager ragManager, int maxConcurrent) {
        this(port, registry, plugins, ragManager, maxConcurrent, new ServerConfig());
    }

    public ApiServer(int port, ModelRegistry registry, PluginManager plugins,
                     RagManager ragManager, int maxConcurrent, ServerConfig cfg) {
        this.port               = port;
        this.registry           = registry;
        this.plugins            = plugins != null ? plugins : PluginManager.EMPTY;
        this.ragManager         = ragManager;
        this.maxConcurrent      = maxConcurrent;
        this.maxBodyBytes       = cfg.maxBodyBytes;
        this.maxOutputTokens    = cfg.maxOutputTokens;
        this.rateLimiter        = cfg.rateLimitPerMinute > 0
                                  ? new IpRateLimiter(cfg.rateLimitPerMinute) : null;
        this.requestExecutor    = createExecutor();
        this.inferenceSemaphore = new Semaphore(maxConcurrent);
        this.contextPool        = new ContextPool(maxConcurrent);
        this.prettyGson         = new GsonBuilder().setPrettyPrinting().create();
        this.compactGson        = new Gson();
    }

    // ── Virtual thread detection and executor creation ────────────────────────

    /**
     * Returns true if the current JVM supports virtual threads (Java 21+).
     * Uses a method lookup rather than {@code Runtime.version()} so it compiles
     * on Java 11 without source-level changes.
     */
    private static boolean detectVirtualThreads() {
        try {
            Executors.class.getMethod("newVirtualThreadPerTaskExecutor");
            return true;
        } catch (NoSuchMethodException e) {
            return false;
        }
    }

    /**
     * Creates a virtual-thread-per-task executor on Java 21+, or a cached
     * platform-thread pool on older JVMs. Both behave identically from
     * Undertow's perspective: {@link BlockingHandler} dispatches each request
     * to the executor and the handler runs to completion on that thread.
     */
    private static ExecutorService createExecutor() {
        if (VIRTUAL_THREADS_AVAILABLE) {
            try {
                Method m = Executors.class.getMethod("newVirtualThreadPerTaskExecutor");
                return (ExecutorService) m.invoke(null);
            } catch (Exception e) {
                LOG.warn("Failed to create virtual thread executor, falling back to platform threads", e);
            }
        }
        // Fallback: unbounded cached pool of daemon platform threads.
        return Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "jllm-worker");
            t.setDaemon(true);
            return t;
        });
    }

    public void start() throws Exception {
        HttpHandler router = Handlers.routing()
            // root info
            .get("/",                     b(this::handleRoot))
            // ── Observability ─────────────────────────────────────────────────
            .get("/health",               b(this::handleHealth))
            .get("/metrics",              b(this::handleMetrics))
            // ── Ollama API ────────────────────────────────────────────────────
            .get("/api/tags",             b(this::handleTags))
            .get("/api/ps",               b(this::handlePs))
            .post("/api/show",            b(this::handleShow))
            .post("/api/generate",        b(this::handleGenerate))
            .post("/api/chat",            b(this::handleChat))
            .post("/api/embeddings",      b(this::handleEmbeddings))
            // ── Model management API ──────────────────────────────────────────
            .post("/api/pull",            b(this::handleApiPull))
            .post("/api/create",          b(this::handleApiCreate))
            .post("/api/copy",            b(this::handleApiCopy))
            .post("/api/delete",          b(this::handleApiDelete))
            .delete("/api/delete",        b(this::handleApiDelete))
            .post("/api/add",             b(this::handleApiAdd))
            // ── Plugin management ─────────────────────────────────────────────
            .get("/api/plugins",          b(this::handlePlugins))
            .post("/api/plugins/reload",  b(this::handlePluginsReload))
            // ── OpenAI API ───────────────────────────────────────────────────
            .get("/v1/models",            b(this::handleV1Models))
            .post("/v1/chat/completions", b(this::handleV1ChatCompletions))
            .post("/v1/completions",      b(this::handleV1Completions))
            .post("/v1/embeddings",       b(this::handleV1Embeddings));

        Undertow server = Undertow.builder()
            .addHttpListener(port, "0.0.0.0")
            .setHandler(withCors(router))
            .build();
        server.start();
        startPluginWatcher();

        System.out.printf("Listening on http://localhost:%d%n", port);
        System.out.println();
        if (VIRTUAL_THREADS_AVAILABLE) {
            System.out.println("Threads          : Virtual (Java 21+) — lightweight, unmount between tokens");
        } else {
            System.out.println("Threads          : Platform (cached pool) — upgrade to Java 21+ for Virtual Threads");
        }
        System.out.printf("Max concurrent   : %d inference slot(s) (--max-concurrent to change)%n", maxConcurrent);
        System.out.printf("Context pool     : enabled — up to %d idle context(s) per model config%n", maxConcurrent);
        System.out.printf("Max body         : %s (--max-body to change)%n", formatBytes(maxBodyBytes));
        System.out.printf("Max output tokens: %s (--max-tokens to change)%n",
            maxOutputTokens > 0 ? String.valueOf(maxOutputTokens) : "unlimited");
        System.out.printf("Rate limit       : %s%n",
            rateLimiter != null ? rateLimiter.limit + " req/min per IP (--rate-limit to change)"
                                : "disabled (--rate-limit to enable)");
        System.out.println();
        System.out.println("Observability:");
        System.out.printf("  GET  /health                http://localhost:%d/health%n", port);
        System.out.printf("  GET  /metrics               http://localhost:%d/metrics%n", port);
        System.out.println();
        System.out.println("Ollama-compatible:");
        System.out.printf("  GET  /api/tags              http://localhost:%d/api/tags%n", port);
        System.out.printf("  GET  /api/ps                http://localhost:%d/api/ps%n", port);
        System.out.printf("  POST /api/show              http://localhost:%d/api/show%n", port);
        System.out.printf("  POST /api/generate          http://localhost:%d/api/generate%n", port);
        System.out.printf("  POST /api/chat              http://localhost:%d/api/chat%n", port);
        System.out.printf("  POST /api/embeddings        http://localhost:%d/api/embeddings%n", port);
        System.out.println();
        System.out.println("Model management:");
        System.out.printf("  POST /api/pull              http://localhost:%d/api/pull%n", port);
        System.out.printf("  POST /api/create            http://localhost:%d/api/create%n", port);
        System.out.printf("  POST /api/copy              http://localhost:%d/api/copy%n", port);
        System.out.printf("  POST /api/delete            http://localhost:%d/api/delete%n", port);
        System.out.printf("  POST /api/add               http://localhost:%d/api/add%n", port);
        System.out.println();
        System.out.println("Plugins:");
        System.out.printf("  GET  /api/plugins           http://localhost:%d/api/plugins%n", port);
        System.out.printf("  POST /api/plugins/reload    http://localhost:%d/api/plugins/reload%n", port);
        System.out.printf("  %d tool(s), %d interceptor(s) loaded from %s%n",
            plugins.getTools().size(), plugins.getInterceptors().size(),
            plugins.getPluginDir() != null ? plugins.getPluginDir() : "(none)");
        System.out.printf("  Directory watch: %s (auto-reloads on plugin JAR changes)%n",
            plugins.getPluginDir() != null && Files.isDirectory(plugins.getPluginDir()) ? "enabled" : "disabled");
        System.out.println();
        System.out.println("OpenAI-compatible:");
        System.out.printf("  GET  /v1/models             http://localhost:%d/v1/models%n", port);
        System.out.printf("  POST /v1/chat/completions   http://localhost:%d/v1/chat/completions%n", port);
        System.out.printf("  POST /v1/completions        http://localhost:%d/v1/completions%n", port);
        System.out.printf("  POST /v1/embeddings         http://localhost:%d/v1/embeddings%n", port);
        System.out.println();
        System.out.println("Press Ctrl+C to stop.");

        Thread.currentThread().join();
    }

    // ── Undertow helpers ──────────────────────────────────────────────────────

    /**
     * Wrap a handler so it runs on {@link #requestExecutor} (virtual threads on
     * Java 21+, platform threads otherwise) rather than on Undertow's IO thread.
     * All handlers do blocking work (JNI, SSE writes), so they must all be wrapped.
     */
    private HttpHandler b(HttpHandler h) {
        return exchange -> {
            if (exchange.isInIoThread()) {
                exchange.dispatch(requestExecutor, () -> {
                    exchange.startBlocking();
                    long start = System.nanoTime();
                    try {
                        if (checkRateLimit(exchange)) h.handleRequest(exchange);
                    } catch (HandledRequestException ignored) {
                        // response already sent inside the handler
                    } catch (Exception e) {
                        LOG.error("Handler error", e);
                    } finally {
                        metrics.record(pathToLabel(exchange.getRequestPath()),
                                       exchange.getStatusCode(), System.nanoTime() - start);
                    }
                });
            } else {
                exchange.startBlocking();
                long start = System.nanoTime();
                try {
                    if (checkRateLimit(exchange)) h.handleRequest(exchange);
                } catch (HandledRequestException ignored) {
                } catch (Exception e) {
                    LOG.error("Handler error", e);
                } finally {
                    metrics.record(pathToLabel(exchange.getRequestPath()),
                                   exchange.getStatusCode(), System.nanoTime() - start);
                }
            }
        };
    }

    private static String pathToLabel(String path) {
        return path.replaceFirst("^/", "").replace('/', '_').replace('-', '_');
    }

    // ── Plugin directory watcher ─────────────────────────────────────────────

    /**
     * Starts a daemon thread that watches the plugin directory and calls
     * {@link PluginManager#load()} whenever a JAR is added, changed, or removed —
     * so dropping in a rebuilt plugin JAR during development takes effect without
     * restarting the server. No-op if there is no plugin directory to watch.
     */
    private void startPluginWatcher() {
        Path dir = plugins.getPluginDir();
        if (dir == null || !Files.isDirectory(dir)) return;

        Thread t = new Thread(() -> watchPluginDir(dir), "jllm-plugin-watcher");
        t.setDaemon(true);
        t.start();
    }

    private void watchPluginDir(Path dir) {
        try (WatchService ws = dir.getFileSystem().newWatchService()) {
            dir.register(ws, StandardWatchEventKinds.ENTRY_CREATE,
                              StandardWatchEventKinds.ENTRY_MODIFY,
                              StandardWatchEventKinds.ENTRY_DELETE);
            while (true) {
                WatchKey key = ws.take(); // blocks until something changes

                // Plugin JARs are often written in several steps (copy, then
                // rename); wait briefly and drain the burst before reloading
                // so we don't reload mid-copy on a half-written file.
                Thread.sleep(300);
                key.pollEvents();
                boolean valid = key.reset();

                try {
                    plugins.load();
                    LOG.info("Plugin directory changed — reloaded: {} tool(s), {} interceptor(s)",
                            plugins.getTools().size(), plugins.getInterceptors().size());
                } catch (Exception e) {
                    LOG.warn("Plugin auto-reload failed: {}", e.getMessage());
                }

                if (!valid) break; // directory itself is no longer accessible
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (IOException e) {
            LOG.warn("Plugin directory watcher stopped: {}", e.getMessage());
        }
    }

    /**
     * Check the per-IP rate limit. Returns true if the request is allowed,
     * sends HTTP 429 and returns false if the IP is over the limit.
     */
    private boolean checkRateLimit(HttpServerExchange ex) throws Exception {
        if (rateLimiter == null) return true;
        String ip = ex.getSourceAddress().getAddress().getHostAddress();
        if (rateLimiter.allow(ip)) return true;
        metrics.recordRateLimited();
        ex.getResponseHeaders()
            .put(Headers.CONTENT_TYPE, "application/json")
            .put(new HttpString("Retry-After"), "60");
        ex.setStatusCode(429);
        byte[] body = ("{\"error\":\"rate limit exceeded: max " + rateLimiter.limit
            + " requests/min per IP\"}")
            .getBytes(StandardCharsets.UTF_8);
        ex.setResponseContentLength(body.length);
        ex.getOutputStream().write(body);
        return false;
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
        ContextPool.PoolStats ps = contextPool.stats();
        sendJson(ex, 200, map(
            "name",    "local-llm",
            "version", "1.0.0",
            "ctx_pool", map(
                "hits",      ps.hits,
                "misses",    ps.misses,
                "hit_rate",  String.format("%.1f%%", ps.hitRate()),
                "idle",      ps.totalIdle,
                "evictions", ps.evictions
            )
        ));
    }

    // ── Ollama: GET /api/ps ───────────────────────────────────────────────────

    private void handlePs(HttpServerExchange ex) throws Exception {
        ContextPool.PoolStats ps = contextPool.stats();

        // Idle-context-pool stats, grouped by model name — a model can have
        // pooled contexts under several nCtx/nThreads configs at once.
        Map<String, List<Map<String, Object>>> configsByModel = new LinkedHashMap<>();
        Map<String, Integer> idleByModel = new LinkedHashMap<>();
        for (ContextPool.KeyStats ks : ps.byKey) {
            // key format: "modelName|nCtx|nThreads"
            String[] parts = ks.key.split("\\|", 3);
            String modelName = parts[0];
            Map<String, Object> cfgEntry = new LinkedHashMap<>();
            cfgEntry.put("num_ctx",       parts.length > 1 ? parts[1] : "?");
            cfgEntry.put("num_threads",   parts.length > 2 ? parts[2] : "?");
            cfgEntry.put("idle_contexts", ks.idleContexts);
            configsByModel.computeIfAbsent(modelName, k -> new ArrayList<>()).add(cfgEntry);
            idleByModel.merge(modelName, ks.idleContexts, Integer::sum);
        }

        // One row per model actually resident in memory (loadedModels is the
        // source of truth). Using the context pool alone would miss a model
        // whose contexts are all currently checked out (e.g. every slot busy
        // serving a request) — idle_contexts would read 0, but the model is
        // very much loaded and in use.
        List<Map<String, Object>> models = new ArrayList<>();
        for (String modelName : loadedModels.keySet()) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("name",          modelName);
            entry.put("idle_contexts", idleByModel.getOrDefault(modelName, 0));
            entry.put("configs",       configsByModel.getOrDefault(modelName, Collections.emptyList()));
            ModelConfig cfg = registry.get(modelName).orElse(null);
            if (cfg != null) entry.put("size_bytes", cfg.getSizeBytes());
            models.add(entry);
        }

        List<Map<String, Object>> schedulers = new ArrayList<>();
        batchSchedulers.forEach((key, sched) -> {
            String[] parts = key.split("\\|", 3);
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("name",              parts[0]);
            entry.put("active_sequences",  sched.activeSequences());
            entry.put("pending_requests",  sched.pendingRequests());
            entry.put("num_ctx",           parts.length > 1 ? parts[1] : "?");
            entry.put("num_threads",       parts.length > 2 ? parts[2] : "?");
            schedulers.add(entry);
        });

        sendJson(ex, 200, map(
            "models",    models,
            "batch_schedulers", schedulers,
            "pool_stats", map(
                "hits",      ps.hits,
                "misses",    ps.misses,
                "hit_rate",  String.format("%.1f%%", ps.hitRate()),
                "total_idle", ps.totalIdle,
                "evictions", ps.evictions
            )
        ));
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
        opts.numPredict = capTokens(opts.numPredict);
        String ragCollection = req.has("rag_collection") ? req.get("rag_collection").getAsString() : null;
        String effectiveSystem = ragEnhancedSystem(ragCollection, prompt, cfg.getSystemPrompt());
        String effectivePrompt = plugins.applyInterceptors(withSystemPrompt(effectiveSystem, prompt));
        int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
        int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;

        try {
            if (opts.stream) {
                beginNdjson(ex);
                try (OutputStream os = ex.getOutputStream()) {
                    streamTokens(model, modelName, effectivePrompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                        piece -> writeNdjson(os, ollamaGenerateChunk(modelName, piece, false)));
                    writeNdjson(os, ollamaGenerateChunk(modelName, "", true));
                }
            } else {
                StringBuilder sb = new StringBuilder();
                streamTokens(model, modelName, effectivePrompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                    sb::append);
                sendJson(ex, 200, ollamaGenerateChunk(modelName, sb.toString(), true));
            }
        } catch (Exception e) {
            LOG.error("generate failed for '{}'", modelName, e);
            if (!ex.isResponseStarted()) sendError(ex, 500, "generation failed: " + e.getMessage());
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
        opts.numPredict = capTokens(opts.numPredict);
        String ragCollection = req.has("rag_collection") ? req.get("rag_collection").getAsString() : null;
        String ragQuery = lastUserMessage(messages);
        String effectiveSystem = ragEnhancedSystem(ragCollection, ragQuery, cfg.getSystemPrompt());
        String prompt = plugins.applyInterceptors(chatMlPrompt(messages, effectiveSystem));
        int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
        int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;

        try {
            if (opts.stream) {
                beginNdjson(ex);
                try (OutputStream os = ex.getOutputStream()) {
                    streamTokens(model, modelName, prompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                        piece -> writeNdjson(os, ollamaChatChunk(modelName, piece, false)));
                    writeNdjson(os, ollamaChatChunk(modelName, "", true));
                }
            } else {
                StringBuilder sb = new StringBuilder();
                streamTokens(model, modelName, prompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                    sb::append);
                sendJson(ex, 200, ollamaChatChunk(modelName, sb.toString(), true));
            }
        } catch (Exception e) {
            LOG.error("chat failed for '{}'", modelName, e);
            if (!ex.isResponseStarted()) sendError(ex, 500, "chat failed: " + e.getMessage());
        }
    }

    // ── Ollama: POST /api/embeddings ──────────────────────────────────────────

    private void handleEmbeddings(HttpServerExchange ex) throws Exception {
        JsonObject req   = parseJson(ex);
        String modelName = req.has("model") ? req.get("model").getAsString() : null;
        // Ollama accepts both "prompt" (single string) and "input" (string or array)
        String text = null;
        if (req.has("prompt") && !req.get("prompt").isJsonNull()) {
            text = req.get("prompt").getAsString();
        } else if (req.has("input") && !req.get("input").isJsonNull()) {
            if (req.get("input").isJsonArray()) {
                // For array input, embed first element only (Ollama v1 behaviour)
                text = req.getAsJsonArray("input").get(0).getAsString();
            } else {
                text = req.get("input").getAsString();
            }
        }
        if (modelName == null || modelName.isEmpty()) { sendError(ex, 400, "model is required"); return; }
        if (text == null || text.isEmpty())            { sendError(ex, 400, "prompt or input is required"); return; }

        ModelConfig cfg = requireModel(ex, modelName); if (cfg == null) return;
        LlamaModel model = loadModel(ex, cfg);         if (model == null) return;

        if (!LlamaModel.isNativeLibraryAvailable()) {
            sendError(ex, 501, "embeddings require the native JNI library");
            return;
        }

        try {
            int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
            int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;
            float[] embd = model.embed(text, nCtx, nThreads);

            List<Double> embdList = new ArrayList<>(embd.length);
            for (float v : embd) embdList.add((double) v);

            sendJson(ex, 200, map("model", modelName, "embedding", embdList));
        } catch (Exception e) {
            LOG.error("embeddings failed for '{}'", modelName, e);
            if (!ex.isResponseStarted()) sendError(ex, 500, "embeddings failed: " + e.getMessage());
        }
    }

    // ── OpenAI: POST /v1/embeddings ───────────────────────────────────────────

    private void handleV1Embeddings(HttpServerExchange ex) throws Exception {
        JsonObject req   = parseJson(ex);
        String modelName = req.has("model") ? req.get("model").getAsString() : null;
        // "input" can be a string or array of strings; embed first (or only) element
        String text = null;
        if (req.has("input") && !req.get("input").isJsonNull()) {
            if (req.get("input").isJsonArray()) {
                JsonArray arr = req.getAsJsonArray("input");
                if (arr.size() > 0) text = arr.get(0).getAsString();
            } else {
                text = req.get("input").getAsString();
            }
        }
        if (modelName == null || modelName.isEmpty()) { sendError(ex, 400, "model is required"); return; }
        if (text == null || text.isEmpty())            { sendError(ex, 400, "input is required"); return; }

        ModelConfig cfg = requireModel(ex, modelName); if (cfg == null) return;
        LlamaModel model = loadModel(ex, cfg);         if (model == null) return;

        if (!LlamaModel.isNativeLibraryAvailable()) {
            sendError(ex, 501, "embeddings require the native JNI library");
            return;
        }

        try {
            int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
            int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;
            float[] embd = model.embed(text, nCtx, nThreads);

            List<Double> embdList = new ArrayList<>(embd.length);
            for (float v : embd) embdList.add((double) v);

            Map<String, Object> dataEntry = new LinkedHashMap<>();
            dataEntry.put("object",    "embedding");
            dataEntry.put("embedding", embdList);
            dataEntry.put("index",     0);

            sendJson(ex, 200, map(
                "object", "list",
                "data",   Collections.singletonList(dataEntry),
                "model",  modelName,
                "usage",  map("prompt_tokens", 0, "total_tokens", 0)
            ));
        } catch (Exception e) {
            LOG.error("v1/embeddings failed for '{}'", modelName, e);
            if (!ex.isResponseStarted()) sendError(ex, 500, "embeddings failed: " + e.getMessage());
        }
    }

    // ── Management: POST /api/pull ────────────────────────────────────────────

    /**
     * Download a GGUF from HuggingFace and register it.
     * Request: {@code {"name":"owner/repo/file.gguf","as":"alias","token":"hf_..."}}
     * Response: NDJSON progress stream, final line: {@code {"status":"success","name":"..."}}
     */
    private void handleApiPull(HttpServerExchange ex) throws Exception {
        JsonObject req = parseJson(ex);
        if (!req.has("name")) { sendError(ex, 400, "\"name\" is required (format: owner/repo/file.gguf)"); return; }
        String ref     = req.get("name").getAsString();
        String hfToken = req.has("token") ? req.get("token").getAsString() : System.getenv("HF_TOKEN");
        String alias   = req.has("as") && !req.get("as").isJsonNull() ? req.get("as").getAsString() : null;

        // Parse owner/repo/file.gguf
        int fs = ref.indexOf('/'), ss = fs >= 0 ? ref.indexOf('/', fs + 1) : -1;
        if (fs < 0 || ss < 0) { sendError(ex, 400, "expected format: owner/repo/filename.gguf"); return; }
        String owner    = ref.substring(0, fs);
        String repo     = ref.substring(fs + 1, ss);
        String filePath = ref.substring(ss + 1);
        if (owner.isEmpty() || repo.isEmpty() || filePath.isEmpty()) {
            sendError(ex, 400, "owner, repo and filename must not be empty"); return;
        }

        String fileName = filePath.contains("/") ? filePath.substring(filePath.lastIndexOf('/') + 1) : filePath;
        String name = alias != null ? alias
            : (fileName.toLowerCase(Locale.ROOT).endsWith(".gguf")
               ? fileName.substring(0, fileName.length() - 5) : fileName);
        Path dest = ModelRegistry.getManagedModelsDir().resolve(fileName);

        beginNdjson(ex);
        try (OutputStream os = ex.getOutputStream()) {
            HuggingFaceClient hf = new HuggingFaceClient(hfToken);

            if (Files.exists(dest)) {
                writeNdjson(os, map("status", "already downloaded", "filename", fileName));
            } else {
                writeNdjson(os, map("status", "downloading", "filename", fileName,
                    "repo", owner + "/" + repo));
                final long[] lastPct = {-1};
                final long[] lastMb  = {0};
                try {
                    hf.download(owner, repo, filePath, "main", dest, (dl, total) -> {
                        try {
                            if (total > 0) {
                                long pct = dl * 100 / total;
                                if (pct != lastPct[0]) {
                                    lastPct[0] = pct;
                                    writeNdjson(os, map("status", "downloading",
                                        "completed", dl, "total", total));
                                }
                            } else {
                                long mb = dl / (10L * 1024 * 1024); // report every 10 MB
                                if (mb != lastMb[0]) {
                                    lastMb[0] = mb;
                                    writeNdjson(os, map("status", "downloading", "completed", dl));
                                }
                            }
                        } catch (Exception ignored) {}
                    });
                } catch (Exception e) {
                    writeNdjson(os, map("status", "error", "error", e.getMessage()));
                    return;
                }
                writeNdjson(os, map("status", "download complete", "bytes", Files.size(dest)));
            }

            // Detect and download remaining shards for split models
            SplitGguf.Split split = SplitGguf.detect(dest);
            if (split != null && split.totalShards > 1) {
                writeNdjson(os, map("status", "split model detected", "shards", split.totalShards));
                List<String> remotePaths = SplitGguf.remoteShardPaths(filePath, split.totalShards);
                for (int i = 0; i < split.totalShards; i++) {
                    Path shardDest = split.shards.get(i);
                    if (shardDest.toAbsolutePath().equals(dest.toAbsolutePath())) continue;
                    if (Files.exists(shardDest)) {
                        writeNdjson(os, map("status", "shard already downloaded", "shard", i + 1));
                        continue;
                    }
                    final int shardNum = i + 1;
                    writeNdjson(os, map("status", "downloading shard",
                        "shard", shardNum, "of", split.totalShards));
                    try {
                        hf.download(owner, repo, remotePaths.get(i), "main", shardDest, null);
                    } catch (Exception e) {
                        writeNdjson(os, map("status", "error",
                            "error", "shard " + shardNum + " failed: " + e.getMessage()));
                        return;
                    }
                }
                // Re-detect after all shards are present
                split = SplitGguf.detect(dest);
            }

            writeNdjson(os, map("status", "registering", "name", name));
            ModelConfig model = buildModelConfig(name, dest, split);
            registry.add(model);
            writeNdjson(os, map("status", "success", "name", name, "size", model.getSizeBytes()));
        } catch (Exception e) {
            LOG.error("api/pull failed", e);
        }
    }

    // ── Management: POST /api/create ─────────────────────────────────────────

    /**
     * Create a model from an inline Modelfile string.
     * Request: {@code {"name":"my-model","modelfile":"FROM /path/to/model.gguf\nSYSTEM ..."}}
     * The FROM directive may also name a registered model (by name) instead of a file path.
     */
    private void handleApiCreate(HttpServerExchange ex) throws Exception {
        JsonObject req = parseJson(ex);
        if (!req.has("name") || !req.has("modelfile")) {
            sendError(ex, 400, "\"name\" and \"modelfile\" are required"); return;
        }
        String name    = req.get("name").getAsString();
        String content = req.get("modelfile").getAsString();
        boolean stream = !req.has("stream") || req.get("stream").getAsBoolean();

        ModelConfig model = new ModelConfig();
        model.setName(name);
        model.setFormat("gguf");
        model.setAddedAt(Instant.now().toString());
        Modelfile.apply(content, model);

        if (model.getPath() == null || model.getPath().isEmpty()) {
            sendError(ex, 400, "Modelfile must contain a FROM directive"); return;
        }

        Path modelPath = Paths.get(model.getPath());
        if (!Files.exists(modelPath)) {
            // FROM may reference a registered model name rather than a file path
            ModelConfig src = registry.get(model.getPath()).orElse(null);
            if (src == null) { sendError(ex, 400, "file not found: " + model.getPath()); return; }
            model.setPath(src.getPath());
            model.setShards(src.getShards());
            model.setSizeBytes(src.getSizeBytes());
            model.setGgufArchitecture(src.getGgufArchitecture());
            model.setGgufQuantization(src.getGgufQuantization());
            model.setGgufParameterCount(src.getGgufParameterCount());
            model.setGgufContextLength(src.getGgufContextLength());
            model.setGgufBlockCount(src.getGgufBlockCount());
            model.setGgufEmbeddingLength(src.getGgufEmbeddingLength());
        } else {
            SplitGguf.Split split = SplitGguf.detect(modelPath);
            applyShardInfoToModel(model, modelPath, split);
            applyGgufMetadata(model, Paths.get(model.getPath()));
        }

        registry.add(model);

        if (stream) {
            beginNdjson(ex);
            try (OutputStream os = ex.getOutputStream()) {
                writeNdjson(os, map("status", "success"));
            }
        } else {
            sendJson(ex, 200, map("name", name));
        }
    }

    // ── Management: POST /api/copy ────────────────────────────────────────────

    /**
     * Copy a registry entry to a new name (no file duplication).
     * Request: {@code {"source":"src-name","destination":"dst-name"}}
     */
    private void handleApiCopy(HttpServerExchange ex) throws Exception {
        JsonObject req = parseJson(ex);
        if (!req.has("source") || !req.has("destination")) {
            sendError(ex, 400, "\"source\" and \"destination\" are required"); return;
        }
        String source      = req.get("source").getAsString();
        String destination = req.get("destination").getAsString();

        ModelConfig src = registry.get(source).orElse(null);
        if (src == null) { sendError(ex, 404, "model '" + source + "' not found"); return; }

        // Clone via JSON round-trip — ModelConfig is a plain POJO with no cycles.
        ModelConfig dst = compactGson.fromJson(compactGson.toJson(src), ModelConfig.class);
        dst.setName(destination);
        dst.setAddedAt(Instant.now().toString());
        registry.add(dst);

        sendJson(ex, 200, map("source", source, "destination", destination));
    }

    // ── Management: POST /api/delete  or  DELETE /api/delete ─────────────────

    /**
     * Unregister a model.
     * Request: {@code {"name":"model-name"}} — add {@code "purge":true} to also delete the GGUF file(s).
     */
    private void handleApiDelete(HttpServerExchange ex) throws Exception {
        JsonObject req = parseJson(ex);
        if (!req.has("name")) { sendError(ex, 400, "\"name\" is required"); return; }
        String  name  = req.get("name").getAsString();
        boolean purge = req.has("purge") && req.get("purge").getAsBoolean();

        ModelConfig cfg = registry.get(name).orElse(null);
        if (cfg == null) { sendError(ex, 404, "model '" + name + "' not found"); return; }

        // Evict in-memory caches so the next request doesn't resurrect the entry.
        loadedModels.remove(name);
        batchSchedulers.entrySet().removeIf(e -> e.getKey().startsWith(name + "|"));

        registry.remove(name);

        if (purge) {
            List<String> toDelete = cfg.isSplit()
                ? cfg.getShards() : Collections.singletonList(cfg.getPath());
            long freed = 0;
            for (String s : toDelete) {
                Path p = Paths.get(s);
                if (Files.exists(p)) { freed += Files.size(p); Files.delete(p); }
            }
            sendJson(ex, 200, map("deleted", name, "purged", true, "freed_bytes", freed));
        } else {
            sendJson(ex, 200, map("deleted", name));
        }
    }

    // ── Management: POST /api/add (jllm-specific) ─────────────────────────────

    /**
     * Register a model file already present on the server's filesystem.
     * <pre>
     * {
     *   "name": "my-model",
     *   "path": "/absolute/path/to/model.gguf",
     *   "managed": false,          // if true, copy to ~/.local-llm/models/ first
     *   "system": "...",           // optional Modelfile parameters
     *   "temperature": 0.7,
     *   "num_ctx": 4096,
     *   "num_predict": 512,
     *   "num_threads": 4,
     *   "num_gpu_layers": 35
     * }
     * </pre>
     */
    private void handleApiAdd(HttpServerExchange ex) throws Exception {
        JsonObject req = parseJson(ex);
        if (!req.has("name") || !req.has("path")) {
            sendError(ex, 400, "\"name\" and \"path\" are required"); return;
        }
        String  name    = req.get("name").getAsString();
        String  rawPath = req.get("path").getAsString();
        boolean managed = req.has("managed") && req.get("managed").getAsBoolean();

        Path src = Paths.get(rawPath);
        if (!Files.exists(src)) { sendError(ex, 400, "file not found: " + rawPath); return; }

        if (managed) {
            Files.createDirectories(ModelRegistry.getManagedModelsDir());
            Path dst = ModelRegistry.getManagedModelsDir().resolve(src.getFileName());
            if (!dst.toAbsolutePath().equals(src.toAbsolutePath())) {
                Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);
                src = dst;
            }
        }

        SplitGguf.Split split = SplitGguf.detect(src);
        ModelConfig model = buildModelConfig(name, src, split);

        if (req.has("system")        && !req.get("system").isJsonNull())
            model.setSystemPrompt(req.get("system").getAsString());
        if (req.has("temperature")   && !req.get("temperature").isJsonNull())
            model.setTemperature(req.get("temperature").getAsFloat());
        if (req.has("num_ctx")       && !req.get("num_ctx").isJsonNull())
            model.setNumCtx(req.get("num_ctx").getAsInt());
        if (req.has("num_predict")   && !req.get("num_predict").isJsonNull())
            model.setNumPredict(req.get("num_predict").getAsInt());
        if (req.has("num_threads")   && !req.get("num_threads").isJsonNull())
            model.setNumThreads(req.get("num_threads").getAsInt());
        if (req.has("num_gpu_layers") && !req.get("num_gpu_layers").isJsonNull())
            model.setNumGpuLayers(req.get("num_gpu_layers").getAsInt());

        registry.add(model);

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("name",  name);
        r.put("path",  model.getPath());
        r.put("size",  model.getSizeBytes());
        if (model.isSplit())                    r.put("shards",       model.getShards().size());
        if (model.getGgufQuantization() != null) r.put("quantization", model.getGgufQuantization());
        if (model.getGgufParameterCount() != null) r.put("parameters", model.getGgufParameterCount());
        sendJson(ex, 200, r);
    }

    // ── Management helpers ────────────────────────────────────────────────────

    /**
     * Build a fully-populated {@link ModelConfig} for {@code name} from a GGUF at
     * {@code primaryPath}, handling split models via the pre-detected {@code split}.
     */
    private static ModelConfig buildModelConfig(String name, Path primaryPath,
                                                SplitGguf.Split split) throws Exception {
        ModelConfig model = new ModelConfig();
        model.setName(name);
        model.setFormat("gguf");
        model.setAddedAt(Instant.now().toString());
        applyShardInfoToModel(model, primaryPath, split);
        applyGgufMetadata(model, Paths.get(model.getPath()));
        return model;
    }

    private static void applyShardInfoToModel(ModelConfig model, Path primaryPath,
                                              SplitGguf.Split split) throws Exception {
        if (split != null && split.totalShards > 1) {
            model.setPath(split.first().toString());
            model.setShards(split.shards.stream().map(Path::toString).collect(Collectors.toList()));
            long total = 0;
            for (Path s : split.shards) { try { total += Files.size(s); } catch (Exception ignored) {} }
            model.setSizeBytes(total);
        } else {
            model.setPath(primaryPath.toString());
            model.setShards(null);
            try { model.setSizeBytes(Files.size(primaryPath)); } catch (Exception ignored) {}
        }
    }

    private static void applyGgufMetadata(ModelConfig model, Path path) {
        try {
            GgufReader.GgufMetadata m = GgufReader.read(path);
            model.setGgufArchitecture(m.architecture);
            model.setGgufQuantization(m.quantization);
            model.setGgufParameterCount(m.parameterCount);
            model.setGgufContextLength(m.contextLength);
            model.setGgufBlockCount(m.blockCount);
            model.setGgufEmbeddingLength(m.embeddingLength);
        } catch (Exception ignored) {}
    }

    // ── Plugins: GET /api/plugins ─────────────────────────────────────────────

    private void handlePlugins(HttpServerExchange ex) throws Exception {
        sendJson(ex, 200, pluginsSummary());
    }

    // ── Plugins: POST /api/plugins/reload ─────────────────────────────────────

    /**
     * Rescans the plugin directory and reloads all tools/interceptors in place.
     * Lets a developer drop a new/updated plugin JAR into the plugin directory
     * and pick it up without restarting the server (and losing loaded models,
     * the context pool, etc).
     */
    private void handlePluginsReload(HttpServerExchange ex) throws Exception {
        plugins.load();
        LOG.info("Plugins reloaded via API: {} tool(s), {} interceptor(s)",
                plugins.getTools().size(), plugins.getInterceptors().size());
        sendJson(ex, 200, pluginsSummary());
    }

    private Map<String, Object> pluginsSummary() {
        List<Map<String, Object>> tools = new ArrayList<>();
        for (LlmTool t : plugins.getTools()) {
            tools.add(map("name", t.getName(), "description", t.getDescription(),
                           "source", plugins.getSourceJar(t)));
        }
        List<Map<String, Object>> interceptors = new ArrayList<>();
        for (PromptInterceptor ic : plugins.getInterceptors()) {
            interceptors.add(map("priority", ic.getPriority(), "source", plugins.getSourceJar(ic)));
        }
        return map(
            "plugin_dir",   plugins.getPluginDir() != null ? plugins.getPluginDir().toString() : null,
            "tools",        tools,
            "interceptors", interceptors
        );
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

        // ── tool calling ──────────────────────────────────────────────────────
        JsonArray tools = req.has("tools") && !req.get("tools").isJsonNull()
            ? req.getAsJsonArray("tools") : null;
        // tool_choice can be a string ("auto","none","required") or object; treat object as "auto"
        String toolChoice = "auto";
        if (req.has("tool_choice") && !req.get("tool_choice").isJsonNull()) {
            toolChoice = req.get("tool_choice").isJsonPrimitive()
                ? req.get("tool_choice").getAsString() : "auto";
        }
        boolean useTools = tools != null && tools.size() > 0 && !"none".equals(toolChoice);

        ModelConfig cfg = requireModel(ex, modelName); if (cfg == null) return;
        LlamaModel model = loadModel(ex, cfg);         if (model == null) return;

        opts.applyModelDefaults(cfg);
        opts.numPredict = capTokens(opts.numPredict);
        String ragCollection = req.has("rag_collection") ? req.get("rag_collection").getAsString() : null;
        String effectiveSystem = ragEnhancedSystem(ragCollection, lastUserMessage(messages), cfg.getSystemPrompt());

        // When tools are present inject instructions so the model knows to use <tool_call> tags.
        if (useTools) effectiveSystem = buildToolSystemPrompt(tools, effectiveSystem);

        String prompt = plugins.applyInterceptors(
            useTools ? chatMlPromptWithTools(messages, effectiveSystem)
                     : chatMlPrompt(messages, effectiveSystem));
        int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
        int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;
        String id    = "chatcmpl-" + shortUuid();
        long created = Instant.now().getEpochSecond();

        try {
            if (useTools) {
                // Buffer the full response to detect <tool_call> tags before replying.
                StringBuilder sb = new StringBuilder();
                streamTokens(model, modelName, prompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                    sb::append);
                String response = sb.toString().trim();
                ToolCallResult tcr = extractToolCall(response);

                if (tcr != null) {
                    if (opts.stream) {
                        beginSse(ex);
                        try (OutputStream os = ex.getOutputStream()) {
                            writeSse(os, openAiToolCallChunk(id, modelName, created, tcr));
                            writeSse(os, openAiSseFinishChunk(id, modelName, created, "tool_calls"));
                            os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                            os.flush();
                        }
                    } else {
                        sendJson(ex, 200, openAiToolCallResponse(id, modelName, created, tcr));
                    }
                } else {
                    // Model didn't call a tool — return as a normal completion.
                    if (opts.stream) {
                        beginSse(ex);
                        try (OutputStream os = ex.getOutputStream()) {
                            writeSse(os, openAiChatChunk(id, modelName, created, "", "assistant", null));
                            writeSse(os, openAiChatChunk(id, modelName, created, response, null, null));
                            writeSse(os, openAiChatChunk(id, modelName, created, "", null, "stop"));
                            os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                            os.flush();
                        }
                    } else {
                        sendJson(ex, 200, openAiChatResponse(id, modelName, created, response));
                    }
                }
            } else {
                // Normal flow — no tools.
                if (opts.stream) {
                    beginSse(ex);
                    try (OutputStream os = ex.getOutputStream()) {
                        writeSse(os, openAiChatChunk(id, modelName, created, "", "assistant", null));
                        streamTokens(model, modelName, prompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                            piece -> writeSse(os, openAiChatChunk(id, modelName, created, piece, null, null)));
                        writeSse(os, openAiChatChunk(id, modelName, created, "", null, "stop"));
                        os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                        os.flush();
                    }
                } else {
                    StringBuilder sb = new StringBuilder();
                    streamTokens(model, modelName, prompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                        sb::append);
                    sendJson(ex, 200, openAiChatResponse(id, modelName, created, sb.toString()));
                }
            }
        } catch (Exception e) {
            LOG.error("v1/chat/completions failed for '{}'", modelName, e);
            if (!ex.isResponseStarted()) sendError(ex, 500, "generation failed: " + e.getMessage());
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
        opts.numPredict = capTokens(opts.numPredict);
        String ragCollection = req.has("rag_collection") ? req.get("rag_collection").getAsString() : null;
        String ragSystem = ragEnhancedSystem(ragCollection, prompt, cfg.getSystemPrompt());
        String effectivePrompt = plugins.applyInterceptors(withSystemPrompt(ragSystem, prompt));
        int nCtx     = cfg.getNumCtx()     != null ? cfg.getNumCtx()     : DEFAULT_N_CTX;
        int nThreads = cfg.getNumThreads() != null ? cfg.getNumThreads() : DEFAULT_N_THREADS;
        String id    = "cmpl-" + shortUuid();
        long created = Instant.now().getEpochSecond();

        try {
            if (opts.stream) {
                beginSse(ex);
                try (OutputStream os = ex.getOutputStream()) {
                    streamTokens(model, modelName, effectivePrompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                        piece -> writeSse(os, openAiCompletionChunk(id, modelName, created, piece, null)));
                    writeSse(os, openAiCompletionChunk(id, modelName, created, "", "stop"));
                    os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                    os.flush();
                }
            } else {
                StringBuilder sb = new StringBuilder();
                streamTokens(model, modelName, effectivePrompt, opts.numPredict, opts.temperature, nCtx, nThreads,
                    sb::append);
                sendJson(ex, 200, openAiCompletionResponse(id, modelName, created, sb.toString()));
            }
        } catch (Exception e) {
            LOG.error("v1/completions failed for '{}'", modelName, e);
            if (!ex.isResponseStarted()) sendError(ex, 500, "generation failed: " + e.getMessage());
        }
    }

    // ── RAG helpers ───────────────────────────────────────────────────────────

    /**
     * Search the named RAG collection for {@code query} and return a system prompt
     * that prepends the retrieved context block in front of {@code baseSystem}.
     * Returns {@code baseSystem} unchanged if RAG is not configured or finds nothing.
     */
    private String ragEnhancedSystem(String collection, String query, String baseSystem) {
        if (ragManager == null || collection == null || collection.isEmpty()) return baseSystem;
        try {
            List<RagResult> hits = ragManager.search(collection, query);
            String ctx = RagManager.buildContextBlock(hits);
            if (ctx == null) return baseSystem;
            return (baseSystem != null && !baseSystem.isEmpty())
                ? ctx + "\n\n" + baseSystem : ctx;
        } catch (Exception e) {
            LOG.warn("RAG search failed for collection '{}': {}", collection, e.getMessage());
            return baseSystem;
        }
    }

    /** Extract the text of the last user-role message from a messages array. */
    private static String lastUserMessage(JsonArray messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            JsonObject msg = messages.get(i).getAsJsonObject();
            if ("user".equals(msg.get("role").getAsString())) {
                return msg.get("content").getAsString();
            }
        }
        return "";
    }

    // ── HTTP I/O helpers ──────────────────────────────────────────────────────

    private JsonObject parseJson(HttpServerExchange ex) throws Exception {
        // Fast-reject: Content-Length header already exceeds limit
        long cl = ex.getRequestContentLength();
        if (cl > maxBodyBytes) {
            sendError(ex, 413, "request body too large: " + cl
                + " bytes (limit: " + formatBytes(maxBodyBytes) + ")");
            metrics.recordBodyTooLarge();
            throw new HandledRequestException();
        }
        // Hard cap on actual bytes read (covers chunked transfer without Content-Length)
        byte[] raw = readBodyLimited(ex.getInputStream(), maxBodyBytes);
        if (raw == null) {
            sendError(ex, 413, "request body too large (limit: " + formatBytes(maxBodyBytes) + ")");
            metrics.recordBodyTooLarge();
            throw new HandledRequestException();
        }
        return prettyGson.fromJson(new String(raw, StandardCharsets.UTF_8), JsonObject.class);
    }

    /**
     * Read at most {@code limit} bytes from {@code in}.
     * Returns the bytes on success, or {@code null} if the stream exceeds {@code limit}.
     */
    private static byte[] readBodyLimited(InputStream in, int limit) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream(Math.min(limit, 65536));
        byte[] chunk = new byte[8192];
        int total = 0, n;
        while ((n = in.read(chunk)) != -1) {
            total += n;
            if (total > limit) return null;
            baos.write(chunk, 0, n);
        }
        return baos.toByteArray();
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

    /** Clamp token count to server-configured maximum (0 = no cap). */
    private int capTokens(int n) {
        return (maxOutputTokens > 0) ? Math.min(n, maxOutputTokens) : n;
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
            int nGpuLayers = cfg.getNumGpuLayers() != null ? cfg.getNumGpuLayers() : 0;
            return loadedModels.computeIfAbsent(cfg.getName(), n -> new LlamaModel(cfg.getPath(), nGpuLayers));
        } catch (Exception e) {
            LOG.error("failed to load model '{}'", cfg.getName(), e);
            sendError(ex, 500, "failed to load model: " + e.getMessage());
            return null;
        }
    }

    // ── Inference dispatch ────────────────────────────────────────────────────

    @FunctionalInterface
    private interface TokenSink {
        void accept(String piece) throws Exception;
    }

    /**
     * Stream tokens for {@code prompt} into {@code sink}, using the
     * {@link BatchScheduler} when the JNI library is available and falling
     * back to the {@link ContextPool} path otherwise.
     */
    private void streamTokens(LlamaModel model, String modelName,
                               String prompt, int nPredict, float temperature,
                               int nCtx, int nThreads,
                               TokenSink sink) throws Exception {
        TokenSink countingSink = piece -> { metrics.recordToken(); sink.accept(piece); };
        if (LlamaModel.isNativeLibraryAvailable()) {
            BatchScheduler sched = getOrCreateScheduler(model, modelName, nCtx, nThreads);
            int[] tokens = model.tokenize(prompt, true, true);
            BatchScheduler.Sequence seq = sched.submit(tokens, nPredict, temperature);
            String piece;
            while ((piece = seq.nextPiece()) != null) {
                countingSink.accept(piece);
            }
        } else {
            inferenceSemaphore.acquire();
            LlamaContext ctx = contextPool.acquire(model, modelName, nCtx, nThreads);
            try {
                try (LlamaContext.TokenStream ts = ctx.generateTokens(prompt, nPredict, temperature)) {
                    for (String piece : ts) countingSink.accept(piece);
                }
            } finally {
                contextPool.release(modelName, nCtx, nThreads, ctx);
                inferenceSemaphore.release();
            }
        }
    }

    private BatchScheduler getOrCreateScheduler(LlamaModel model, String modelName,
                                                 int nCtx, int nThreads) {
        String key = modelName + "|" + nCtx + "|" + nThreads;
        return batchSchedulers.computeIfAbsent(key,
            k -> new BatchScheduler(model, nCtx, nThreads, maxConcurrent));
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

    // ── Tool calling helpers ──────────────────────────────────────────────────

    /**
     * Parsed result of a {@code <tool_call>} tag found in model output.
     * Carries a unique call-id for the client to correlate tool results.
     */
    private static final class ToolCallResult {
        final String id;
        final String name;
        final String arguments; // compact JSON string

        ToolCallResult(String name, String arguments) {
            this.id        = "call_" + UUID.randomUUID().toString().replace("-", "").substring(0, 8);
            this.name      = name;
            this.arguments = arguments;
        }
    }

    /**
     * Build a system-prompt extension that tells the model to use {@code <tool_call>} tags
     * and lists the available functions with their JSON schemas.
     */
    private static String buildToolSystemPrompt(JsonArray tools, String baseSystem) {
        StringBuilder sb = new StringBuilder();
        if (baseSystem != null && !baseSystem.isEmpty()) sb.append(baseSystem).append("\n\n");
        sb.append("You have access to the following functions. ");
        sb.append("To call a function, respond with ONLY:\n");
        sb.append("<tool_call>{\"name\":\"FUNCTION_NAME\",\"arguments\":{...}}</tool_call>\n");
        sb.append("Do not include any other text when calling a function.\n\n");
        sb.append("Available functions (JSON Schema):\n");
        for (int i = 0; i < tools.size(); i++) {
            JsonObject t = tools.get(i).getAsJsonObject();
            // Each tool entry is {"type":"function","function":{...}} per OpenAI spec.
            sb.append(t.has("function") ? t.get("function").toString() : t.toString()).append("\n");
        }
        return sb.toString().stripTrailing();
    }

    /**
     * Like {@link #chatMlPrompt} but also handles {@code role:"tool"} result messages
     * and {@code role:"assistant"} messages that carry {@code tool_calls} (conversation
     * history replay from a previous turn).
     */
    private static String chatMlPromptWithTools(JsonArray messages, String systemPrompt) {
        boolean hasSystem = false;
        for (int i = 0; i < messages.size(); i++) {
            if ("system".equals(messages.get(i).getAsJsonObject().get("role").getAsString())) {
                hasSystem = true; break;
            }
        }
        StringBuilder sb = new StringBuilder();
        if (!hasSystem && systemPrompt != null && !systemPrompt.isEmpty()) {
            sb.append("<|im_start|>system\n").append(systemPrompt).append("<|im_end|>\n");
        }
        for (int i = 0; i < messages.size(); i++) {
            JsonObject msg  = messages.get(i).getAsJsonObject();
            String role     = msg.get("role").getAsString();
            String content  = msg.has("content") && !msg.get("content").isJsonNull()
                              ? msg.get("content").getAsString() : "";

            if ("tool".equals(role)) {
                // Tool result injected as a user turn so the model can see the outcome.
                String name = msg.has("name") ? msg.get("name").getAsString() : "tool";
                sb.append("<|im_start|>user\n[Tool result for ").append(name).append("]: ")
                  .append(content).append("<|im_end|>\n");

            } else if ("assistant".equals(role) && msg.has("tool_calls")
                       && !msg.get("tool_calls").isJsonNull()) {
                // Replay an earlier assistant turn that called a function.
                sb.append("<|im_start|>assistant\n");
                JsonArray tcs = msg.getAsJsonArray("tool_calls");
                for (int j = 0; j < tcs.size(); j++) {
                    JsonObject tc = tcs.get(j).getAsJsonObject();
                    JsonObject fn = tc.getAsJsonObject("function");
                    String fnName = fn.get("name").getAsString();
                    String fnArgs = fn.has("arguments") ? fn.get("arguments").getAsString() : "{}";
                    sb.append("<tool_call>{\"name\":\"").append(fnName)
                      .append("\",\"arguments\":").append(fnArgs).append("}</tool_call>");
                }
                sb.append("<|im_end|>\n");

            } else {
                sb.append("<|im_start|>").append(role).append("\n")
                  .append(content).append("<|im_end|>\n");
            }
        }
        return sb.append("<|im_start|>assistant\n").toString();
    }

    /**
     * Detect and parse a {@code <tool_call>} tag in {@code text}.
     * Accepts both {@code "arguments"} and {@code "args"} as the parameter key.
     * Returns {@code null} if no valid tool call is found.
     */
    private static ToolCallResult extractToolCall(String text) {
        Matcher m = TOOL_CALL_RE.matcher(text);
        if (!m.find()) return null;
        try {
            JsonObject obj  = new Gson().fromJson(m.group(1).trim(), JsonObject.class);
            String name     = obj.get("name").getAsString();
            JsonElement args = obj.has("arguments") ? obj.get("arguments")
                             : obj.has("args")      ? obj.get("args") : null;
            String argsJson = (args != null && args.isJsonObject()) ? args.toString() : "{}";
            return new ToolCallResult(name, argsJson);
        } catch (Exception e) {
            return null;
        }
    }

    /** Non-streaming response body when the model calls a function. */
    private static Map<String, Object> openAiToolCallResponse(
            String id, String model, long created, ToolCallResult tcr) {
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", tcr.name);
        fn.put("arguments", tcr.arguments);

        Map<String, Object> tc = new LinkedHashMap<>();
        tc.put("id",       tcr.id);
        tc.put("type",     "function");
        tc.put("function", fn);

        Map<String, Object> msg = new LinkedHashMap<>();
        msg.put("role",       "assistant");
        msg.put("content",    null);
        msg.put("tool_calls", Collections.singletonList(tc));

        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index",         0);
        choice.put("message",       msg);
        choice.put("finish_reason", "tool_calls");

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("id",      id);
        r.put("object",  "chat.completion");
        r.put("created", created);
        r.put("model",   model);
        r.put("choices", Collections.singletonList(choice));
        r.put("usage",   map("prompt_tokens", 0, "completion_tokens", 0, "total_tokens", 0));
        return r;
    }

    /** First SSE chunk carrying the function-call delta (name + all arguments in one shot). */
    private static Map<String, Object> openAiToolCallChunk(
            String id, String model, long created, ToolCallResult tcr) {
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name",      tcr.name);
        fn.put("arguments", tcr.arguments);

        Map<String, Object> tc = new LinkedHashMap<>();
        tc.put("index",    0);
        tc.put("id",       tcr.id);
        tc.put("type",     "function");
        tc.put("function", fn);

        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("role",       "assistant");
        delta.put("content",    null);
        delta.put("tool_calls", Collections.singletonList(tc));

        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index",         0);
        choice.put("delta",         delta);
        choice.put("finish_reason", null);

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("id",      id);
        r.put("object",  "chat.completion.chunk");
        r.put("created", created);
        r.put("model",   model);
        r.put("choices", Collections.singletonList(choice));
        return r;
    }

    /** Terminal SSE chunk with an empty delta and the given finish_reason. */
    private static Map<String, Object> openAiSseFinishChunk(
            String id, String model, long created, String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index",         0);
        choice.put("delta",         Collections.emptyMap());
        choice.put("finish_reason", finishReason);

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("id",      id);
        r.put("object",  "chat.completion.chunk");
        r.put("created", created);
        r.put("model",   model);
        r.put("choices", Collections.singletonList(choice));
        return r;
    }

    // ── GET /health ───────────────────────────────────────────────────────────

    private void handleHealth(HttpServerExchange ex) throws Exception {
        Runtime rt = Runtime.getRuntime();
        long heapUsed  = rt.totalMemory() - rt.freeMemory();
        long heapMax   = rt.maxMemory();
        long uptimeSec = (System.currentTimeMillis() - metrics.startMs) / 1000;
        int  slotsAvail = inferenceSemaphore.availablePermits();
        ContextPool.PoolStats ps = contextPool.stats();

        sendJson(ex, 200, map(
            "status",          slotsAvail == 0 && maxConcurrent > 0 ? "busy" : "ok",
            "uptime_seconds",  uptimeSec,
            "loaded_models",   loadedModels.size(),
            "inference_slots", map(
                "active",    maxConcurrent - slotsAvail,
                "available", slotsAvail,
                "max",       maxConcurrent
            ),
            "context_pool", map(
                "idle",     ps.totalIdle,
                "hit_rate", String.format("%.1f%%", ps.hitRate())
            ),
            "jvm", map(
                "heap_used_bytes", heapUsed,
                "heap_max_bytes",  heapMax,
                "heap_used_mb",    heapUsed / (1024 * 1024),
                "heap_max_mb",     heapMax  / (1024 * 1024),
                "threads",         ManagementFactory.getThreadMXBean().getThreadCount()
            )
        ));
    }

    // ── GET /metrics (Prometheus text format) ─────────────────────────────────

    private void handleMetrics(HttpServerExchange ex) throws Exception {
        StringBuilder sb = new StringBuilder(8192);
        long uptimeSec = (System.currentTimeMillis() - metrics.startMs) / 1000;
        Runtime rt = Runtime.getRuntime();
        long heapUsed  = rt.totalMemory() - rt.freeMemory();
        long heapMax   = rt.maxMemory();
        ContextPool.PoolStats ps = contextPool.stats();
        int slotsAvail = inferenceSemaphore.availablePermits();

        // uptime
        promMetric(sb, "jllm_uptime_seconds", "gauge", "Server uptime in seconds", uptimeSec);

        // tokens / rejections
        promMetric(sb, "jllm_tokens_generated_total", "counter",
            "Total output tokens generated across all requests", metrics.tokensTotal.sum());
        promMetric(sb, "jllm_rate_limit_rejected_total", "counter",
            "Requests rejected by the per-IP rate limiter", metrics.rateLimitedTotal.sum());
        promMetric(sb, "jllm_body_too_large_total", "counter",
            "Requests rejected because the body exceeded --max-body", metrics.bodyTooLargeTotal.sum());

        // inference slots
        promMetric(sb, "jllm_inference_slots_active", "gauge",
            "Inference slots currently occupied", maxConcurrent - slotsAvail);
        promMetric(sb, "jllm_inference_slots_total", "gauge",
            "Total configured inference slots (--max-concurrent)", maxConcurrent);

        // context pool
        promMetric(sb, "jllm_context_pool_hits_total",      "counter",
            "Context pool cache hits (KV cache reused)",      ps.hits);
        promMetric(sb, "jllm_context_pool_misses_total",    "counter",
            "Context pool cache misses (new context created)", ps.misses);
        promMetric(sb, "jllm_context_pool_evictions_total", "counter",
            "Context pool evictions",                          ps.evictions);
        promMetric(sb, "jllm_context_pool_idle",            "gauge",
            "Idle contexts in the pool",                       ps.totalIdle);

        // loaded models
        promMetric(sb, "jllm_models_loaded", "gauge",
            "Number of model weights currently loaded in memory", loadedModels.size());

        // per-model batch scheduler gauges
        if (!batchSchedulers.isEmpty()) {
            sb.append("# HELP jllm_batch_active_sequences Active sequences in the batch scheduler\n");
            sb.append("# TYPE jllm_batch_active_sequences gauge\n");
            sb.append("# HELP jllm_batch_pending_requests Pending requests in the batch scheduler\n");
            sb.append("# TYPE jllm_batch_pending_requests gauge\n");
            batchSchedulers.forEach((key, sched) -> {
                String model = key.split("\\|", 2)[0].replace("\"", "\\\"");
                sb.append("jllm_batch_active_sequences{model=\"").append(model).append("\"} ")
                  .append(sched.activeSequences()).append('\n');
                sb.append("jllm_batch_pending_requests{model=\"").append(model).append("\"} ")
                  .append(sched.pendingRequests()).append('\n');
            });
        }

        // JVM memory / threads
        promMetric(sb, "jllm_jvm_heap_used_bytes", "gauge",
            "JVM heap bytes currently used", heapUsed);
        promMetric(sb, "jllm_jvm_heap_max_bytes", "gauge",
            "JVM heap maximum bytes (-Xmx)", heapMax);
        promMetric(sb, "jllm_jvm_threads", "gauge",
            "JVM thread count", ManagementFactory.getThreadMXBean().getThreadCount());

        // per-endpoint request counters + duration histogram
        if (!metrics.endpoints.isEmpty()) {
            sb.append("# HELP jllm_requests_total HTTP requests total by endpoint and status class\n");
            sb.append("# TYPE jllm_requests_total counter\n");
            metrics.endpoints.forEach((ep, ec) -> {
                sb.append("jllm_requests_total{endpoint=\"").append(ep).append("\",status=\"2xx\"} ")
                  .append(ec.ok.sum()).append('\n');
                sb.append("jllm_requests_total{endpoint=\"").append(ep).append("\",status=\"4xx\"} ")
                  .append(ec.clientErr.sum()).append('\n');
                sb.append("jllm_requests_total{endpoint=\"").append(ep).append("\",status=\"5xx\"} ")
                  .append(ec.serverErr.sum()).append('\n');
            });

            sb.append("# HELP jllm_request_duration_seconds HTTP request latency histogram\n");
            sb.append("# TYPE jllm_request_duration_seconds histogram\n");
            metrics.endpoints.forEach((ep, ec) -> {
                for (int i = 0; i < MetricsCollector.HIST_LE.length; i++) {
                    sb.append("jllm_request_duration_seconds_bucket{endpoint=\"").append(ep)
                      .append("\",le=\"").append(MetricsCollector.HIST_LE[i]).append("\"} ")
                      .append(ec.buckets[i].sum()).append('\n');
                }
                sb.append("jllm_request_duration_seconds_sum{endpoint=\"").append(ep).append("\"} ")
                  .append(String.format("%.6f", ec.durationNsSum.sum() / 1_000_000_000.0)).append('\n');
                sb.append("jllm_request_duration_seconds_count{endpoint=\"").append(ep).append("\"} ")
                  .append(ec.total.sum()).append('\n');
            });
        }

        byte[] body = sb.toString().getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().put(Headers.CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8");
        ex.setStatusCode(200);
        ex.setResponseContentLength(body.length);
        ex.getOutputStream().write(body);
    }

    private static void promMetric(StringBuilder sb, String name, String type,
                                   String help, long value) {
        sb.append("# HELP ").append(name).append(' ').append(help).append('\n');
        sb.append("# TYPE ").append(name).append(' ').append(type).append('\n');
        sb.append(name).append(' ').append(value).append('\n');
    }

    // ── Misc helpers ──────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private static <K, V> Map<K, V> map(Object... pairs) {
        Map<Object, Object> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < pairs.length; i += 2) m.put(pairs[i], pairs[i + 1]);
        return (Map<K, V>) m;
    }

    private static String formatBytes(long bytes) {
        if (bytes >= 1024 * 1024) return String.format("%.0f MB", bytes / (1024.0 * 1024));
        if (bytes >= 1024)        return String.format("%.0f KB", bytes / 1024.0);
        return bytes + " B";
    }

    private static long epochOf(String isoInstant) {
        if (isoInstant == null) return Instant.now().getEpochSecond();
        try { return Instant.parse(isoInstant).getEpochSecond(); }
        catch (Exception e) { return Instant.now().getEpochSecond(); }
    }

    private static String shortUuid() {
        return UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    // ── Metrics collector ─────────────────────────────────────────────────────

    /** Thread-safe request counters and latency histograms for /metrics output. */
    static final class MetricsCollector {

        // Histogram bucket upper bounds in nanoseconds (last = +Inf sentinel, excluded from HIST_BUCKET_NS)
        static final long[]   HIST_BUCKET_NS = {
            50_000_000L, 100_000_000L, 250_000_000L, 500_000_000L,
            1_000_000_000L, 5_000_000_000L, 30_000_000_000L
        };
        static final String[] HIST_LE = {"0.05","0.1","0.25","0.5","1","5","30","+Inf"};

        final long startMs = System.currentTimeMillis();

        static final class EndpointCounter {
            final LongAdder total        = new LongAdder();
            final LongAdder ok           = new LongAdder();   // 2xx
            final LongAdder clientErr    = new LongAdder();   // 4xx
            final LongAdder serverErr    = new LongAdder();   // 5xx
            final LongAdder durationNsSum = new LongAdder();
            // Cumulative histogram buckets; length == HIST_LE.length (+Inf always 1:1 with total)
            final LongAdder[] buckets    = new LongAdder[HIST_LE.length];
            { for (int i = 0; i < buckets.length; i++) buckets[i] = new LongAdder(); }

            void record(int status, long durationNs) {
                total.increment();
                if      (status >= 500) serverErr.increment();
                else if (status >= 400) clientErr.increment();
                else                    ok.increment();
                durationNsSum.add(durationNs);
                // Prometheus histograms are cumulative: bucket[i] = count of requests <= threshold
                for (int i = 0; i < HIST_BUCKET_NS.length; i++) {
                    if (durationNs <= HIST_BUCKET_NS[i]) buckets[i].increment();
                }
                buckets[buckets.length - 1].increment(); // +Inf = total
            }
        }

        final ConcurrentHashMap<String, EndpointCounter> endpoints = new ConcurrentHashMap<>();
        final LongAdder tokensTotal       = new LongAdder();
        final LongAdder rateLimitedTotal  = new LongAdder();
        final LongAdder bodyTooLargeTotal = new LongAdder();

        void record(String endpoint, int status, long durationNs) {
            endpoints.computeIfAbsent(endpoint, k -> new EndpointCounter()).record(status, durationNs);
        }

        void recordToken()        { tokensTotal.increment(); }
        void recordRateLimited()  { rateLimitedTotal.increment(); }
        void recordBodyTooLarge() { bodyTooLargeTotal.increment(); }
    }

    // ── Server configuration ──────────────────────────────────────────────────

    /** Configurable safety limits passed to the ApiServer constructor. */
    public static final class ServerConfig {
        /** Maximum request body size in bytes (default 4 MB). */
        public int maxBodyBytes      = DEFAULT_MAX_BODY_BYTES;
        /** Server-side cap on output tokens; 0 = no cap. */
        public int maxOutputTokens   = DEFAULT_MAX_OUTPUT_TOKENS;
        /** Max requests per minute per client IP; 0 = rate limiting disabled. */
        public int rateLimitPerMinute = DEFAULT_RATE_LIMIT_PER_MIN;
    }

    // ── Rate limiter ──────────────────────────────────────────────────────────

    /**
     * Per-IP fixed-window rate limiter.
     *
     * <p>Each IP gets a one-minute window. The first request in a window starts the
     * clock; subsequent requests within the same window are counted. Requests beyond
     * {@link #limit} in a window are rejected with HTTP 429.
     *
     * <p>The internal map is bounded: when it exceeds 50 000 entries a full sweep
     * evicts all entries whose window has already expired so memory stays bounded
     * even under a large number of distinct client IPs.
     */
    static final class IpRateLimiter {
        final int limit;
        private static final long WINDOW_MS = 60_000L;
        private static final int  EVICT_THRESHOLD = 50_000;

        // Value: long[2] = { windowStartMs, count }
        private final ConcurrentHashMap<String, long[]> state = new ConcurrentHashMap<>();

        IpRateLimiter(int limit) { this.limit = limit; }

        /** Returns true if the request should be allowed, false if rate-limited. */
        boolean allow(String ip) {
            long now = System.currentTimeMillis();
            long[] slot = state.computeIfAbsent(ip, k -> new long[]{now, 0L});
            synchronized (slot) {
                if (now - slot[0] >= WINDOW_MS) {
                    // New window: reset counter and evict stale entries if needed.
                    slot[0] = now;
                    slot[1] = 0;
                    if (state.size() > EVICT_THRESHOLD) evictExpired(now);
                }
                if (slot[1] >= limit) return false;
                slot[1]++;
                return true;
            }
        }

        private void evictExpired(long now) {
            Iterator<Map.Entry<String, long[]>> it = state.entrySet().iterator();
            while (it.hasNext()) {
                long[] s = it.next().getValue();
                if (now - s[0] >= WINDOW_MS) it.remove();
            }
        }

        /** Approximate seconds until the current window resets for any IP. */
        int retryAfterSeconds() { return 60; }
    }

    // ── Sentinel exception ────────────────────────────────────────────────────

    /**
     * Thrown inside a handler after it has already written the error response,
     * so the {@code b()} wrapper knows not to log it as an unexpected error.
     */
    private static final class HandledRequestException extends RuntimeException {
        HandledRequestException() { super(null, null, true, false); }
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
