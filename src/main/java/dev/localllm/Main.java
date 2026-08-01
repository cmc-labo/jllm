package dev.localllm;

import dev.localllm.model.GgufReader;
import dev.localllm.model.JllmfileParser;
import dev.localllm.model.ModelConfig;
import dev.localllm.model.Modelfile;
import dev.localllm.model.ModelRegistry;
import dev.localllm.model.SplitGguf;
import dev.localllm.jni.LlamaModel;
import dev.localllm.jni.LlamaNative;
import dev.localllm.jni.QuantizeType;
import dev.localllm.plugin.LlmTool;
import dev.localllm.plugin.PluginManager;
import dev.localllm.plugin.PromptInterceptor;
import dev.localllm.pull.HuggingFaceClient;
import dev.localllm.rag.RagManager;
import dev.localllm.rag.RagResult;
import dev.localllm.runner.ModelRunner;
import dev.localllm.server.ApiServer;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.InputStream;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

public class Main {

    private static final ModelRegistry registry   = new ModelRegistry();
    private static final RagManager    ragManager = new RagManager(ModelRegistry.getRagDir(), registry);

    // Loaded once at startup; shared across commands.
    private static final PluginManager plugins;

    static {
        PluginManager pm = new PluginManager(ModelRegistry.getPluginsDir());
        pm.load();
        plugins = pm;
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            printUsage();
            return;
        }

        String cmd = args[0];
        switch (cmd) {
            case "list":    cmdList();          break;
            case "add":     cmdAdd(args);       break;
            case "create":  cmdCreate(args);    break;
            case "rm":
            case "remove":  cmdRemove(args);    break;
            case "run":     cmdRun(args);       break;
            case "serve":   cmdServe(args);     break;
            case "rag":     cmdRag(args);       break;
            case "show":    cmdShow(args);      break;
            case "info":    cmdInfo(args);      break;
            case "storage": cmdStorage();       break;
            case "plugins": cmdPlugins(args);   break;
            case "ps":      cmdPs(args);        break;
            case "update":  cmdUpdate(args);    break;
            case "verify":  cmdVerify(args);    break;
            case "version": cmdVersion();       break;
            case "pull":    cmdPull(args);      break;
            default:
                System.err.println("Unknown command: " + cmd);
                printUsage();
                System.exit(1);
        }
    }

    // ── commands ──────────────────────────────────────────────────────────────

    private static void cmdList() {
        List<ModelConfig> models = registry.list();
        if (models.isEmpty()) {
            System.out.println("No models registered. Use 'add' or 'create' to register one.");
            return;
        }
        System.out.printf("%-25s %-10s %-8s %-7s %-10s %-9s %s%n",
                "NAME", "QUANT", "PARAMS", "SHARDS", "SIZE", "STATUS", "PATH");
        System.out.println("-".repeat(107));
        long totalBytes = 0;
        int missingCount = 0;
        for (ModelConfig m : models) {
            boolean exists = Files.exists(Paths.get(m.getPath()));
            String status = exists ? "ok" : "missing";
            String quant  = m.getGgufQuantization()   != null ? m.getGgufQuantization()             : "-";
            String params = m.getGgufParameterCount() != null ? fmtParamCount(m.getGgufParameterCount()) : "-";
            String shards = m.isSplit() ? String.valueOf(m.getShards().size()) : "-";
            System.out.printf("%-25s %-10s %-8s %-7s %-10s %-9s %s%n",
                m.getName(), quant, params, shards, formatSize(m.getSizeBytes()), status, m.getPath());
            if (exists) totalBytes += m.getSizeBytes();
            else        missingCount++;
        }
        System.out.println("-".repeat(107));
        System.out.printf("%d model(s)  %s total on disk", models.size(), formatSize(totalBytes));
        if (missingCount > 0) System.out.printf("  (%d file(s) missing — run 'storage' for details)", missingCount);
        System.out.println();
    }

    private static void cmdAdd(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: jllm add <name> --path <path> [--binary <path>] [--format <fmt>]");
            System.err.println("                       [--managed] [--no-hash]");
            System.err.println("                       [--quantize <type>] [--quantize-output <path>] [--threads <int>]");
            System.exit(1);
        }

        String       name         = args[1];
        String       path         = null;
        String       binary       = null;
        String       format       = "gguf";
        boolean      managed      = false;
        boolean      noHash       = false;
        QuantizeType quantizeType = null;
        String       quantizeOut  = null;
        int          threads      = 0;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--path":             if (i + 1 < args.length) path        = args[++i]; break;
                case "--binary":           if (i + 1 < args.length) binary      = args[++i]; break;
                case "--format":           if (i + 1 < args.length) format      = args[++i]; break;
                case "--quantize":
                    if (i + 1 < args.length) {
                        quantizeType = QuantizeType.fromString(args[++i]);
                        if (quantizeType == null) {
                            System.err.println("Unknown quantize type: " + args[i]);
                            System.err.println("Valid types: " + QuantizeType.validNames());
                            System.exit(1);
                        }
                    }
                    break;
                case "--quantize-output":  if (i + 1 < args.length) quantizeOut = args[++i]; break;
                case "--threads":
                case "--num-threads":      if (i + 1 < args.length) threads     = Integer.parseInt(args[++i]); break;
                case "--managed":          managed = true; break;
                case "--no-hash":          noHash  = true; break;
            }
        }

        if (path == null) { System.err.println("--path is required"); System.exit(1); }
        if (!Files.exists(Paths.get(path))) {
            System.err.println("Model file not found: " + path);
            System.exit(1);
        }

        if (quantizeType != null) {
            path = runQuantize(path, quantizeType, quantizeOut, managed, threads);
            managed = false; // already placed in final location
        } else if (managed) {
            Path managedDir = ModelRegistry.getManagedModelsDir();
            Files.createDirectories(managedDir);
            Path dest = managedDir.resolve(Paths.get(path).getFileName());
            if (!dest.toAbsolutePath().equals(Paths.get(path).toAbsolutePath())) {
                System.out.println("Copying to managed storage...");
                Files.copy(Paths.get(path), dest, StandardCopyOption.REPLACE_EXISTING);
                path = dest.toString();
                System.out.println("Stored at: " + path);
            }
        }

        if (binary == null) binary = detectLlamaBinary();

        ModelConfig model = new ModelConfig();
        model.setName(name);
        model.setBinary(binary);
        model.setFormat(format);
        model.setAddedAt(Instant.now().toString());
        applyShardInfo(model, path);
        readGgufMetadata(model, model.getPath());
        computeAndSetSha256(model, noHash);
        checkForDuplicates(model.getSha256(), name);

        registry.add(model);
        System.out.println("Registered '" + name + "' (" + formatSize(model.getSizeBytes()) + ")");
        printGgufSummary(model);
        printSha256Summary(model);
        if (binary == null) System.out.println("Warning: llama.cpp binary not found. Use --binary.");
    }

    /**
     * Create (or update) a model from a Modelfile, similar to {@code ollama create}.
     *
     * <pre>
     *   local-llm create &lt;name&gt; -f &lt;Modelfile&gt; [--binary &lt;path&gt;]
     * </pre>
     */
    private static void cmdCreate(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: local-llm create <name> -f <Modelfile> [--binary <path>]");
            System.exit(1);
        }

        String  name         = args[1];
        String  modelfilePath = null;
        String  binary        = null;
        boolean noHash        = false;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "-f":
                case "--file":    if (i + 1 < args.length) modelfilePath = args[++i]; break;
                case "--binary":  if (i + 1 < args.length) binary        = args[++i]; break;
                case "--no-hash": noHash = true; break;
            }
        }

        if (modelfilePath == null) {
            System.err.println("Usage: local-llm create <name> -f <Modelfile> [--binary <path>]");
            System.exit(1);
        }
        if (!Files.exists(Paths.get(modelfilePath))) {
            System.err.println("Modelfile not found: " + modelfilePath);
            System.exit(1);
        }

        String content = Files.readString(Paths.get(modelfilePath));
        ModelConfig model = new ModelConfig();
        model.setName(name);
        model.setFormat("gguf");
        model.setAddedAt(Instant.now().toString());

        if (isYamlFile(modelfilePath)) {
            System.out.println("Parsing Jllmfile (YAML format)...");
            JllmfileParser.apply(content, model);
        } else {
            Modelfile.apply(content, model);
        }

        if (model.getPath() == null || model.getPath().isEmpty()) {
            System.err.println("Modelfile must contain a FROM instruction with the path to the GGUF file.");
            System.exit(1);
        }
        if (!Files.exists(Paths.get(model.getPath()))) {
            System.err.println("Model file specified in FROM not found: " + model.getPath());
            System.exit(1);
        }

        applyShardInfo(model, model.getPath());
        if (binary != null) {
            model.setBinary(binary);
        } else if (model.getBinary() == null) {
            model.setBinary(detectLlamaBinary());
        }
        readGgufMetadata(model, model.getPath());
        computeAndSetSha256(model, noHash);
        checkForDuplicates(model.getSha256(), name);

        registry.add(model);
        System.out.println("Created '" + name + "' (" + formatSize(model.getSizeBytes()) + ")");
        printGgufSummary(model);
        printSha256Summary(model);
        printModelfileParams(model);
    }

    private static void cmdUpdate(String[] args) throws Exception {
        if (args.length < 2) { printUpdateUsage(); System.exit(1); }
        String name = args[1];

        ModelConfig model = registry.get(name).orElse(null);
        if (model == null) {
            System.err.println("Model '" + name + "' not found. Run 'jllm list' to see available models.");
            System.exit(1);
        }

        if (args.length == 2) {
            System.out.println("Current configuration for '" + name + "':");
            printModelfileParams(model);
            System.out.println();
            printUpdateUsage();
            return;
        }

        // Snapshot before changes (for diff output)
        Float   prevTemp      = model.getTemperature();
        Integer prevPredict   = model.getNumPredict();
        Integer prevCtx       = model.getNumCtx();
        Integer prevThreads   = model.getNumThreads();
        Integer prevGpuLayers = model.getNumGpuLayers();
        String  prevSystem    = model.getSystemPrompt();
        String  prevPath      = model.getPath();
        String  prevBinary    = model.getBinary();
        boolean ggufRefreshed = false;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--temperature":
                    if (i + 1 < args.length) model.setTemperature(Float.parseFloat(args[++i]));
                    break;
                case "--max-tokens": case "--num-predict":
                    if (i + 1 < args.length) model.setNumPredict(Integer.parseInt(args[++i]));
                    break;
                case "--ctx": case "--num-ctx":
                    if (i + 1 < args.length) model.setNumCtx(Integer.parseInt(args[++i]));
                    break;
                case "--threads": case "--num-threads":
                    if (i + 1 < args.length) model.setNumThreads(Integer.parseInt(args[++i]));
                    break;
                case "--gpu-layers": case "--num-gpu-layers":
                    if (i + 1 < args.length) model.setNumGpuLayers(Integer.parseInt(args[++i]));
                    break;
                case "--system":
                    if (i + 1 < args.length) model.setSystemPrompt(args[++i]);
                    break;
                case "--no-system":
                    model.setSystemPrompt(null);
                    break;
                case "--path":
                    if (i + 1 < args.length) {
                        String p = args[++i];
                        if (!Files.exists(Paths.get(p))) {
                            System.err.println("File not found: " + p);
                            System.exit(1);
                        }
                        applyShardInfo(model, p);
                        readGgufMetadata(model, model.getPath());
                        computeAndSetSha256(model, false);
                    }
                    break;
                case "--binary":
                    if (i + 1 < args.length) model.setBinary(args[++i]);
                    break;
                case "--refresh-gguf":
                    readGgufMetadata(model, model.getPath());
                    ggufRefreshed = true;
                    break;
                case "--refresh-hash":
                    computeAndSetSha256(model, false);
                    ggufRefreshed = true; // reuse the "extra change" flag
                    break;
                case "--unset":
                    if (i + 1 < args.length) {
                        String param = args[++i];
                        switch (param) {
                            case "temperature":                        model.setTemperature(null);    break;
                            case "max-tokens": case "num-predict":     model.setNumPredict(null);     break;
                            case "ctx":        case "num-ctx":         model.setNumCtx(null);         break;
                            case "threads":    case "num-threads":     model.setNumThreads(null);     break;
                            case "gpu-layers": case "num-gpu-layers":  model.setNumGpuLayers(null);   break;
                            case "system":                             model.setSystemPrompt(null);   break;
                            default:
                                System.err.println("Unknown parameter '" + param + "'.");
                                System.err.println("Unset-able params: temperature  max-tokens  ctx  threads  gpu-layers  system");
                                System.exit(1);
                        }
                    }
                    break;
                default:
                    System.err.println("Unknown flag: " + args[i]);
                    printUpdateUsage();
                    System.exit(1);
            }
        }

        boolean changed = ggufRefreshed
                || !objEq(prevTemp,      model.getTemperature())
                || !objEq(prevPredict,   model.getNumPredict())
                || !objEq(prevCtx,       model.getNumCtx())
                || !objEq(prevThreads,   model.getNumThreads())
                || !objEq(prevGpuLayers, model.getNumGpuLayers())
                || !objEq(prevSystem,    model.getSystemPrompt())
                || !objEq(prevPath,      model.getPath())
                || !objEq(prevBinary,    model.getBinary());

        if (!changed) {
            System.out.println("No changes — configuration already matches the given values.");
            return;
        }

        registry.add(model);
        System.out.println("Updated '" + name + "':");
        printParamDiff("temperature",   prevTemp,      model.getTemperature());
        printParamDiff("num_predict",   prevPredict,   model.getNumPredict());
        printParamDiff("num_ctx",       prevCtx,       model.getNumCtx());
        printParamDiff("num_threads",   prevThreads,   model.getNumThreads());
        printParamDiff("num_gpu_layers",prevGpuLayers, model.getNumGpuLayers());
        printParamDiff("system",        prevSystem,    model.getSystemPrompt());
        printParamDiff("path",          prevPath,      model.getPath());
        printParamDiff("binary",        prevBinary,    model.getBinary());
        if (ggufRefreshed) {
            printGgufSummary(model);
            printSha256Summary(model);
        }
    }

    private static void cmdRemove(String[] args) throws Exception {
        if (args.length < 2) { System.err.println("Usage: jllm rm <name> [--purge]"); System.exit(1); }
        String name = args[1];
        boolean purge = false;
        for (int i = 2; i < args.length; i++) {
            if ("--purge".equals(args[i])) purge = true;
        }

        ModelConfig model = registry.get(name).orElse(null);
        if (model == null) {
            System.err.println("Model '" + name + "' not found.");
            System.exit(1);
        }

        registry.remove(name);
        System.out.println("Removed '" + name + "' from registry.");

        if (purge) {
            List<String> toDelete = model.isSplit()
                    ? model.getShards() : Collections.singletonList(model.getPath());
            long freed = 0;
            for (String s : toDelete) {
                Path p = Paths.get(s);
                if (Files.exists(p)) {
                    long sz = Files.size(p);
                    Files.delete(p);
                    freed += sz;
                    System.out.println("Deleted: " + p);
                } else {
                    System.out.println("Not found (already deleted): " + p);
                }
            }
            if (freed > 0) System.out.println("Freed: " + formatSize(freed));
        }
    }

    private static void cmdRun(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: jllm run <name> [--prompt <text>] [--rag <collection>]");
            System.err.println("                      [--system <prompt>] [--no-system]");
            System.err.println("                      [--temperature <float>] [--max-tokens <int>]");
            System.err.println("                      [--ctx <int>] [--threads <int>]");
            System.exit(1);
        }

        String  promptText          = null;
        String  ragCollection       = null;
        String  overrideSystem      = null;
        boolean clearSystem         = false;
        Float   overrideTemperature = null;
        Integer overrideMaxTokens   = null;
        Integer overrideCtx         = null;
        Integer overrideThreads     = null;
        Integer overrideGpuLayers   = null;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--prompt":
                    if (i + 1 < args.length) promptText = args[++i]; break;
                case "--rag":
                    if (i + 1 < args.length) ragCollection = args[++i]; break;
                case "--system":
                    if (i + 1 < args.length) overrideSystem = args[++i]; break;
                case "--no-system":
                    clearSystem = true; break;
                case "--temperature":
                    if (i + 1 < args.length) overrideTemperature = Float.parseFloat(args[++i]); break;
                case "--max-tokens":
                case "--num-predict":
                    if (i + 1 < args.length) overrideMaxTokens = Integer.parseInt(args[++i]); break;
                case "--ctx":
                case "--num-ctx":
                    if (i + 1 < args.length) overrideCtx = Integer.parseInt(args[++i]); break;
                case "--threads":
                case "--num-threads":
                    if (i + 1 < args.length) overrideThreads = Integer.parseInt(args[++i]); break;
                case "--gpu-layers":
                case "--num-gpu-layers":
                    if (i + 1 < args.length) overrideGpuLayers = Integer.parseInt(args[++i]); break;
                default:
                    System.err.println("Unknown flag: " + args[i]);
                    System.exit(1);
            }
        }

        ModelConfig model = registry.get(args[1]).orElse(null);
        if (model == null) {
            System.err.println("Model '" + args[1] + "' not found. Run 'list' to see available models.");
            System.exit(1);
        }

        // Apply session-level overrides to a shallow copy — registry entry is never modified.
        model = applyRunOverrides(model, overrideTemperature, overrideMaxTokens,
                                  overrideCtx, overrideThreads, overrideGpuLayers,
                                  overrideSystem, clearSystem);

        ModelRunner runner = new ModelRunner(plugins, ragManager, ragCollection);

        // Non-interactive when --prompt is given or stdin is piped (not a terminal).
        boolean isPiped = System.console() == null;
        if (promptText != null || isPiped) {
            String prompt = resolvePrompt(promptText);
            runner.runOnce(model, prompt);
        } else {
            runner.runInteractive(model);
        }
    }

    /**
     * Return the prompt text to use in non-interactive mode.
     * If {@code promptFlag} is non-null, use it directly.
     * Otherwise read all of stdin (for piped use: {@code echo "..." | jllm run}).
     */
    private static String resolvePrompt(String promptFlag) throws Exception {
        if (promptFlag != null) return promptFlag;
        String stdin = new String(System.in.readAllBytes(), StandardCharsets.UTF_8).strip();
        if (stdin.isEmpty()) {
            System.err.println("Error: stdin is empty. Provide input via pipe or --prompt.");
            System.exit(1);
        }
        return stdin;
    }

    /**
     * Return a copy of {@code src} with per-session overrides applied.
     * A {@code null} override means "keep the value from the registered config".
     * {@code clearSystem=true} wipes the system prompt regardless of {@code system}.
     */
    private static ModelConfig applyRunOverrides(ModelConfig src,
            Float temperature, Integer maxTokens, Integer ctx, Integer threads,
            Integer gpuLayers, String system, boolean clearSystem) {
        ModelConfig c = new ModelConfig();
        c.setName(src.getName());
        c.setPath(src.getPath());
        c.setShards(src.getShards());
        c.setBinary(src.getBinary());
        c.setFormat(src.getFormat());
        c.setSizeBytes(src.getSizeBytes());
        c.setAddedAt(src.getAddedAt());
        c.setTemperature(temperature  != null ? temperature  : src.getTemperature());
        c.setNumPredict(maxTokens     != null ? maxTokens    : src.getNumPredict());
        c.setNumCtx(ctx               != null ? ctx          : src.getNumCtx());
        c.setNumThreads(threads       != null ? threads      : src.getNumThreads());
        c.setNumGpuLayers(gpuLayers   != null ? gpuLayers   : src.getNumGpuLayers());
        c.setSystemPrompt(clearSystem ? "" : (system != null ? system : src.getSystemPrompt()));
        return c;
    }

    private static void cmdServe(String[] args) throws Exception {
        int port          = 11434;
        int maxConcurrent = Runtime.getRuntime().availableProcessors();
        ApiServer.ServerConfig cfg = new ApiServer.ServerConfig();

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--port":
                    if (i + 1 < args.length) port = Integer.parseInt(args[++i]);
                    break;
                case "--max-concurrent":
                    if (i + 1 < args.length) maxConcurrent = Integer.parseInt(args[++i]);
                    break;
                case "--max-body":
                    if (i + 1 < args.length) cfg.maxBodyBytes = parseBytes(args[++i]);
                    break;
                case "--max-tokens":
                    if (i + 1 < args.length) cfg.maxOutputTokens = Integer.parseInt(args[++i]);
                    break;
                case "--rate-limit":
                    if (i + 1 < args.length) cfg.rateLimitPerMinute = Integer.parseInt(args[++i]);
                    break;
            }
        }
        System.out.println("Starting local-llm server on http://localhost:" + port);
        new ApiServer(port, registry, plugins, ragManager, maxConcurrent, cfg).start();
    }

    /**
     * Show models currently loaded in memory by a running {@code jllm serve} process.
     * Queries that server's {@code GET /api/ps} over HTTP — there is no shared state
     * between separate {@code jllm} invocations otherwise, since each CLI command runs
     * in its own short-lived JVM.
     */
    private static void cmdPs(String[] args) throws Exception {
        int port = 11434;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--port":
                    if (i + 1 < args.length) port = Integer.parseInt(args[++i]); break;
                default:
                    System.err.println("Unknown flag: " + args[i]);
                    System.err.println("Usage: jllm ps [--port <port>]");
                    System.exit(1);
            }
        }

        HttpClient http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .build();
        HttpRequest req = HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/api/ps"))
                .GET()
                .timeout(Duration.ofSeconds(10))
                .build();

        HttpResponse<String> resp;
        try {
            resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        } catch (ConnectException | HttpConnectTimeoutException e) {
            System.err.println("No jllm serve process found on port " + port + " (connection refused).");
            System.err.println("Start one with: jllm serve" + (port != 11434 ? " --port " + port : ""));
            System.exit(1);
            return;
        }
        if (resp.statusCode() != 200) {
            System.err.println("Error: server returned HTTP " + resp.statusCode() + ": " + resp.body());
            System.exit(1);
            return;
        }

        JsonObject body = JsonParser.parseString(resp.body()).getAsJsonObject();
        JsonArray models = body.getAsJsonArray("models");
        JsonArray schedulers = body.has("batch_schedulers") ? body.getAsJsonArray("batch_schedulers") : new JsonArray();

        if (models == null || models.isEmpty()) {
            System.out.println("No models currently loaded on port " + port + ".");
            return;
        }

        // Sum active_sequences/pending_requests per model name across every
        // (num_ctx, num_threads) tuple that model has an active scheduler for.
        Map<String, Integer> activeByModel  = new LinkedHashMap<>();
        Map<String, Integer> pendingByModel = new LinkedHashMap<>();
        for (int i = 0; i < schedulers.size(); i++) {
            JsonObject s = schedulers.get(i).getAsJsonObject();
            String name = s.get("name").getAsString();
            activeByModel.merge(name, s.get("active_sequences").getAsInt(), Integer::sum);
            pendingByModel.merge(name, s.get("pending_requests").getAsInt(), Integer::sum);
        }

        System.out.printf("%-25s %-10s %-10s %-11s %s%n",
                "NAME", "SIZE", "IDLE CTX", "ACTIVE SEQ", "PENDING");
        System.out.println("-".repeat(67));
        for (int i = 0; i < models.size(); i++) {
            JsonObject m = models.get(i).getAsJsonObject();
            String name = m.get("name").getAsString();
            String size = m.has("size_bytes") ? formatSize(m.get("size_bytes").getAsLong()) : "-";
            int idleCtx = m.has("idle_contexts") ? m.get("idle_contexts").getAsInt() : 0;
            System.out.printf("%-25s %-10s %-10d %-11d %d%n",
                    name, size, idleCtx,
                    activeByModel.getOrDefault(name, 0),
                    pendingByModel.getOrDefault(name, 0));
        }
    }

    /**
     * Parse a byte count that may have an optional suffix: {@code K}, {@code M}, {@code G}
     * (case-insensitive). Examples: {@code 4M} → 4194304, {@code 512K} → 524288.
     */
    private static int parseBytes(String s) {
        s = s.trim();
        char last = s.charAt(s.length() - 1);
        int multiplier = 1;
        if (Character.isLetter(last)) {
            switch (Character.toUpperCase(last)) {
                case 'K': multiplier = 1024;           break;
                case 'M': multiplier = 1024 * 1024;    break;
                case 'G': multiplier = 1024 * 1024 * 1024; break;
            }
            s = s.substring(0, s.length() - 1);
        }
        return Integer.parseInt(s.trim()) * multiplier;
    }

    /**
     * Print the Modelfile (or Jllmfile) representation for a registered model.
     * Use {@code --yaml} to emit YAML/Jllmfile format instead of Modelfile format.
     */
    private static void cmdShow(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: local-llm show <name> [--yaml]");
            System.exit(1);
        }
        String name = args[1];
        boolean yaml = false;
        for (int i = 2; i < args.length; i++) {
            if ("--yaml".equals(args[i])) yaml = true;
        }
        ModelConfig m = registry.get(name).orElse(null);
        if (m == null) { System.err.println("Model '" + name + "' not found."); System.exit(1); }

        if (yaml) {
            System.out.println("# Jllmfile for " + m.getName());
            System.out.print(JllmfileParser.toText(m));
        } else {
            System.out.println("# Modelfile for " + m.getName());
            System.out.print(Modelfile.toText(m));
            printModelfileParams(m);
        }
    }

    private static void cmdInfo(String[] args) {
        if (args.length < 2) { System.err.println("Usage: local-llm info <name>"); System.exit(1); }
        ModelConfig m = registry.get(args[1]).orElse(null);
        if (m == null) { System.err.println("Model '" + args[1] + "' not found."); System.exit(1); }

        System.out.println("Name:     " + m.getName());
        System.out.println("Format:   " + (m.getFormat()  != null ? m.getFormat()  : "-"));
        System.out.println("Path:     " + m.getPath());
        if (m.isSplit()) {
            System.out.println("Shards:   " + m.getShards().size() + " (split model)");
        }
        System.out.println("Binary:   " + (m.getBinary()  != null ? m.getBinary()  : "(not set)"));
        System.out.println("Size:     " + formatSize(m.getSizeBytes()) + (m.isSplit() ? " (total)" : ""));
        System.out.println("Added:    " + (m.getAddedAt() != null ? m.getAddedAt() : "-"));
        System.out.println("SHA-256:  " + (m.getSha256()  != null ? m.getSha256()  : "(not computed — run: jllm update " + m.getName() + " --refresh-hash)"));

        if (m.isSplit()) {
            System.out.println();
            System.out.println("Shard files:");
            for (int i = 0; i < m.getShards().size(); i++) {
                Path sp     = Paths.get(m.getShards().get(i));
                boolean ex  = Files.exists(sp);
                String  sz  = ex ? formatSize(safeFileSize(sp)) : "MISSING";
                System.out.printf("  [%d/%d]  %-10s  %s%n",
                        i + 1, m.getShards().size(), sz, sp.getFileName());
            }
        }

        // GGUF metadata section (only shown when at least one field is present)
        boolean hasGguf = m.getGgufArchitecture() != null || m.getGgufQuantization() != null
                       || m.getGgufParameterCount() != null || m.getGgufContextLength() != null;
        if (hasGguf) {
            System.out.println();
            System.out.println("GGUF Metadata:");
            if (m.getGgufArchitecture()   != null) System.out.println("  Architecture  : " + m.getGgufArchitecture());
            if (m.getGgufQuantization()   != null) System.out.println("  Quantization  : " + m.getGgufQuantization());
            if (m.getGgufParameterCount() != null) System.out.println("  Parameters    : " + fmtParamCount(m.getGgufParameterCount()));
            if (m.getGgufContextLength()  != null) System.out.println("  Context length: " + m.getGgufContextLength());
            if (m.getGgufBlockCount()     != null) System.out.println("  Layers        : " + m.getGgufBlockCount());
            if (m.getGgufEmbeddingLength()!= null) System.out.println("  Embedding dim : " + m.getGgufEmbeddingLength());
        }

        printModelfileParams(m);
    }

    /** Show disk usage for all registered models. */
    private static void cmdStorage() {
        List<ModelConfig> models = registry.list();
        Path managedDir = ModelRegistry.getManagedModelsDir();
        System.out.println("Managed storage dir: " + managedDir);
        System.out.println();

        if (models.isEmpty()) {
            System.out.println("No models registered.");
            return;
        }

        System.out.printf("%-25s %-10s %-10s %s%n", "NAME", "SIZE", "STATUS", "PATH");
        System.out.println("-".repeat(90));

        long totalBytes = 0;
        int missingCount = 0;
        for (ModelConfig m : models) {
            Path filePath = Paths.get(m.getPath());
            boolean exists  = Files.exists(filePath);
            boolean isManaged = filePath.toAbsolutePath().startsWith(managedDir.toAbsolutePath());
            String status = !exists ? "missing" : isManaged ? "managed" : "external";
            System.out.printf("%-25s %-10s %-10s %s%n",
                m.getName(), formatSize(m.getSizeBytes()), status, m.getPath());
            if (exists) totalBytes += m.getSizeBytes();
            else        missingCount++;
        }

        System.out.println("-".repeat(90));
        System.out.printf("Total: %d model(s)  %s on disk%n", models.size(), formatSize(totalBytes));
        if (missingCount > 0) {
            System.out.printf("       %d file(s) missing — remove stale entries with: jllm rm <name>%n", missingCount);
        }
        System.out.println();
        System.out.println("Status legend:");
        System.out.println("  managed   file lives under the managed storage dir (safe to purge via jllm rm --purge)");
        System.out.println("  external  file is registered by path but not copied to managed storage");
        System.out.println("  missing   registered path no longer exists on disk");
    }

    /** Show all currently loaded plugins (tools and interceptors), or reload them with 'reload'. */
    private static void cmdPlugins(String[] args) {
        Path pluginDir = ModelRegistry.getPluginsDir();

        if (args.length > 1 && "reload".equals(args[1])) {
            plugins.load();
            System.out.printf("Reloaded plugin directory: %d tool(s), %d interceptor(s)%n%n",
                    plugins.getTools().size(), plugins.getInterceptors().size());
            System.out.println("Note: this only reloads plugins for this CLI process.");
            System.out.println("A running 'jllm serve' watches the plugin directory itself and");
            System.out.println("auto-reloads on change, or can be told to reload via:");
            System.out.println("  curl -X POST http://localhost:<port>/api/plugins/reload");
            System.out.println("An interactive 'jllm run' session can be reloaded with /reload-plugins.");
            System.out.println();
        }

        System.out.println("Plugin directory: " + pluginDir);
        System.out.println();

        List<LlmTool> tools = plugins.getTools();
        List<PromptInterceptor> interceptors = plugins.getInterceptors();

        if (tools.isEmpty() && interceptors.isEmpty()) {
            System.out.println("No plugins loaded.");
            System.out.println();
            System.out.println("To add a plugin:");
            System.out.println("  1. Implement LlmTool or PromptInterceptor (see examples/plugins/)");
            System.out.println("  2. Add META-INF/services/dev.localllm.plugin.<Interface> to your JAR");
            System.out.println("  3. Drop the JAR into: " + pluginDir);
            System.out.println();
            System.out.println("Caution: plugin JARs run fully trusted, with no sandboxing — they get the");
            System.out.println("same OS-level access as the jllm process itself. Only install plugins you");
            System.out.println("wrote yourself or fully trust. See 'Plugin security' in README.md.");
            return;
        }

        System.out.printf("Tools (%d):%n", tools.size());
        if (tools.isEmpty()) {
            System.out.println("  (none)");
        } else {
            System.out.printf("  %-20s  %-45s  %s%n", "NAME", "DESCRIPTION", "SOURCE JAR");
            System.out.println("  " + "-".repeat(80));
            for (LlmTool t : tools) {
                String desc = t.getDescription();
                if (desc.length() > 44) desc = desc.substring(0, 41) + "...";
                System.out.printf("  %-20s  %-45s  %s%n", t.getName(), desc, plugins.getSourceJar(t));
            }
        }

        System.out.println();
        System.out.printf("Interceptors (%d):%n", interceptors.size());
        if (interceptors.isEmpty()) {
            System.out.println("  (none)");
        } else {
            System.out.printf("  %-8s  %-40s  %s%n", "PRIORITY", "CLASS", "SOURCE JAR");
            System.out.println("  " + "-".repeat(70));
            for (PromptInterceptor ic : interceptors) {
                System.out.printf("  %-8d  %-40s  %s%n",
                        ic.getPriority(), ic.getClass().getSimpleName(), plugins.getSourceJar(ic));
            }
        }
    }

    // ── pull (HuggingFace download) ───────────────────────────────────────────

    private static void cmdPull(String[] args) throws Exception {
        if (args.length < 2) { printPullUsage(); System.exit(1); }

        String       ref          = args[1];
        String       name         = null;
        String       branch       = "main";
        String       token        = System.getenv("HF_TOKEN");
        String       binary       = null;
        boolean      noRegister   = false;
        boolean      noHash       = false;
        QuantizeType quantizeType = null;
        int          threads      = 0;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--name":        if (i + 1 < args.length) name   = args[++i]; break;
                case "--branch":      if (i + 1 < args.length) branch = args[++i]; break;
                case "--token":       if (i + 1 < args.length) token  = args[++i]; break;
                case "--binary":      if (i + 1 < args.length) binary = args[++i]; break;
                case "--no-register": noRegister = true; break;
                case "--no-hash":     noHash     = true; break;
                case "--quantize":
                    if (i + 1 < args.length) {
                        quantizeType = QuantizeType.fromString(args[++i]);
                        if (quantizeType == null) {
                            System.err.println("Unknown quantize type: " + args[i]);
                            System.err.println("Valid types: " + QuantizeType.validNames());
                            System.exit(1);
                        }
                    }
                    break;
                case "--threads":
                case "--num-threads": if (i + 1 < args.length) threads = Integer.parseInt(args[++i]); break;
                default:
                    System.err.println("Unknown flag: " + args[i]);
                    printPullUsage(); System.exit(1);
            }
        }

        // Parse: owner/repo  or  owner/repo/filepath (filepath may contain '/')
        int firstSlash  = ref.indexOf('/');
        int secondSlash = firstSlash >= 0 ? ref.indexOf('/', firstSlash + 1) : -1;
        if (firstSlash < 0) {
            System.err.println("Error: expected <owner>/<repo>[/<file.gguf>] — got: " + ref);
            System.exit(1);
        }
        String owner    = ref.substring(0, firstSlash);
        String repo     = secondSlash > 0 ? ref.substring(firstSlash + 1, secondSlash)
                                           : ref.substring(firstSlash + 1);
        String filePath = secondSlash > 0 ? ref.substring(secondSlash + 1) : null;

        if (owner.isEmpty() || repo.isEmpty()) {
            System.err.println("Error: owner and repo must not be empty.");
            System.exit(1);
        }

        HuggingFaceClient hf = new HuggingFaceClient(token);

        // No filename → list GGUFs; auto-pick if exactly one.
        if (filePath == null || filePath.isEmpty()) {
            System.out.println("Fetching file list from " + owner + "/" + repo + " ...");
            List<HuggingFaceClient.HfFile> ggufFiles;
            try {
                ggufFiles = hf.listGgufFiles(owner, repo);
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                System.exit(1); return;
            }
            if (ggufFiles.isEmpty()) {
                System.err.println("No GGUF files found in " + owner + "/" + repo + ".");
                System.exit(1);
            }
            // For split models, show only shard-1 and skip the rest (they auto-download).
            List<HuggingFaceClient.HfFile> displayFiles = new ArrayList<>();
            for (HuggingFaceClient.HfFile f : ggufFiles) {
                int[] si = SplitGguf.parseShardInfo(f.name);
                if (si == null || si[0] == 1) displayFiles.add(f); // single or shard-1
            }
            if (displayFiles.size() == 1) {
                filePath = displayFiles.get(0).name;
                System.out.println("Found: " + filePath);
                System.out.println();
            } else {
                System.out.println("Multiple GGUF files found. Pick one:");
                System.out.println();
                for (HuggingFaceClient.HfFile f : displayFiles) {
                    int[] si = SplitGguf.parseShardInfo(f.name);
                    if (si != null) {
                        System.out.printf("  jllm pull %s/%s/%s%n", owner, repo, f.name);
                        System.out.printf("       (%d-shard split model — all %d shards download automatically)%n",
                                si[1], si[1]);
                    } else {
                        System.out.printf("  jllm pull %s/%s/%s%n", owner, repo, f.name);
                    }
                }
                System.out.println();
                System.exit(0);
            }
        }

        // Last segment of filePath = the actual filename on disk.
        String fileName = filePath.contains("/")
                ? filePath.substring(filePath.lastIndexOf('/') + 1) : filePath;

        if (name == null) {
            name = fileName.toLowerCase(Locale.ROOT).endsWith(".gguf")
                    ? fileName.substring(0, fileName.length() - 5) : fileName;
        }

        Path dest = ModelRegistry.getManagedModelsDir().resolve(fileName);

        // Download primary shard (or single file).
        downloadShard(hf, owner, repo, filePath, branch, dest);

        // Detect split and download remaining shards.
        SplitGguf.Split split = SplitGguf.detect(dest);
        List<Path> allShards;
        if (split != null && split.totalShards > 1) {
            if (!split.isFirstShard()) {
                System.out.printf("Note: shard %d/%d given; primary shard is: %s%n",
                        split.shardIndex, split.totalShards, split.first().getFileName());
            }
            System.out.printf("Split model: %d shards total%n%n", split.totalShards);
            List<String> remotePaths = SplitGguf.remoteShardPaths(filePath, split.totalShards);
            for (int i = 0; i < split.totalShards; i++) {
                Path shardDest = split.shards.get(i);
                if (shardDest.toAbsolutePath().equals(dest.toAbsolutePath())) continue;
                System.out.printf("[%d/%d] ", i + 1, split.totalShards);
                downloadShard(hf, owner, repo, remotePaths.get(i), branch, shardDest);
            }
            allShards = split.shards;
        } else {
            allShards = Collections.singletonList(dest);
        }

        // Quantize before registering (only for single-file models; split GGUF not supported).
        String primaryPath = allShards.get(0).toString();
        if (quantizeType != null) {
            if (allShards.size() > 1) {
                System.out.println("Warning: --quantize is not supported for split GGUF models; skipping quantization.");
            } else {
                primaryPath = runQuantize(primaryPath, quantizeType, null, false, threads);
            }
        }

        // Register.
        if (!noRegister) {
            if (registry.get(name).isPresent()) {
                System.out.println("Note: '" + name + "' already registered — updating entry.");
            }
            if (binary == null) binary = detectLlamaBinary();
            ModelConfig model = new ModelConfig();
            model.setName(name);
            model.setBinary(binary);
            model.setFormat("gguf");
            model.setAddedAt(Instant.now().toString());
            applyShardInfo(model, primaryPath);
            readGgufMetadata(model, model.getPath());
            computeAndSetSha256(model, noHash);
            checkForDuplicates(model.getSha256(), name);
            registry.add(model);
            System.out.println("Registered: " + name);
            printGgufSummary(model);
            printSha256Summary(model);
            System.out.println("Run with : jllm run " + name);
        }
    }

    /**
     * Quantize {@code inputPath} using the JNI binding and return the path of the
     * output file.  The output path is derived as follows (in priority order):
     * <ol>
     *   <li>If {@code explicitOutput} is non-null, use it as-is.</li>
     *   <li>If {@code toManaged} is true, place the output in managed storage.</li>
     *   <li>Otherwise place it in the same directory as the input, with the
     *       quantization type appended to the stem.</li>
     * </ol>
     */
    private static String runQuantize(String inputPath, QuantizeType type,
                                      String explicitOutput, boolean toManaged,
                                      int threads) throws Exception {
        if (!LlamaModel.isNativeLibraryAvailable()) {
            System.err.println("Error: --quantize requires the JNI native library (libllamajni.so).");
            System.err.println("Build the native library with: bash build.sh (requires CMake + llama.cpp).");
            System.exit(1);
        }

        Path input = Paths.get(inputPath).toAbsolutePath();
        Path output;
        if (explicitOutput != null) {
            output = Paths.get(explicitOutput).toAbsolutePath();
        } else {
            String stem = input.getFileName().toString();
            if (stem.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
                stem = stem.substring(0, stem.length() - 5);
            }
            String outName = stem + "-" + type.name + ".gguf";
            if (toManaged) {
                Path managedDir = ModelRegistry.getManagedModelsDir();
                Files.createDirectories(managedDir);
                output = managedDir.resolve(outName);
            } else {
                output = input.getParent().resolve(outName);
            }
        }

        if (output.toAbsolutePath().equals(input.toAbsolutePath())) {
            System.err.println("Error: quantize output path must differ from input path.");
            System.exit(1);
        }

        System.out.println("Quantizing " + type.name + "...");
        System.out.println("  Input  : " + input);
        System.out.println("  Output : " + output);
        System.out.flush();

        long startMs = System.currentTimeMillis();
        int rc = LlamaNative.quantize(input.toString(), output.toString(), type.ftypeId, threads);
        long elapsed = System.currentTimeMillis() - startMs;

        if (rc != 0 || !Files.exists(output)) {
            System.err.println("Quantization failed (return code " + rc + ").");
            System.exit(1);
        }
        System.out.printf("Quantization complete in %.1f s  (%s → %s)%n",
                elapsed / 1000.0, formatSize(Files.size(input)), formatSize(Files.size(output)));
        return output.toString();
    }

    /** Download a single shard from HuggingFace, skipping if already present. */
    private static void downloadShard(HuggingFaceClient hf,
            String owner, String repo, String remoteFilePath, String branch, Path dest) throws Exception {
        if (Files.exists(dest)) {
            System.out.println("Already downloaded: " + dest.getFileName()
                    + "  (" + formatSize(Files.size(dest)) + ")");
            return;
        }
        System.out.println("Source : " + owner + "/" + repo + "/" + remoteFilePath
                + "  (branch: " + branch + ")");
        System.out.println("Dest   : " + dest);
        long[] startMs = {System.currentTimeMillis()};
        try {
            hf.download(owner, repo, remoteFilePath, branch, dest,
                    (dl, total) -> printPullProgress(dl, total, startMs[0]));
        } catch (Exception e) {
            System.out.println();
            System.err.println("Download failed: " + e.getMessage());
            System.exit(1);
        }
        System.out.println();
        System.out.println("Complete: " + formatSize(Files.size(dest)));
        System.out.println();
    }

    private static void printPullProgress(long downloaded, long total, long startMs) {
        long elapsed = Math.max(1, System.currentTimeMillis() - startMs);
        double bps   = downloaded * 1000.0 / elapsed;

        String downStr  = formatSize(downloaded);
        String speedStr = formatSize((long) bps) + "/s";
        String totalStr = total > 0 ? " / " + formatSize(total) : "";

        String pct = "";
        String bar = "";
        String eta = "";
        if (total > 0) {
            int p = (int) (downloaded * 100 / total);
            pct = String.format("  %3d%%", p);
            int width  = 20;
            int filled = (int) (downloaded * width / total);
            bar = "  [" + "█".repeat(filled) + "░".repeat(width - filled) + "]";
            if (bps > 0) {
                eta = "  ETA " + formatEta((long) ((total - downloaded) / bps));
            }
        }

        System.out.printf("\r  %s%s  %s%s%s%s          ",
                downStr, totalStr, speedStr, bar, pct, eta);
        System.out.flush();
    }

    private static String formatEta(long seconds) {
        if (seconds < 60)   return seconds + "s";
        if (seconds < 3600) return (seconds / 60) + "m " + (seconds % 60) + "s";
        return (seconds / 3600) + "h " + ((seconds % 3600) / 60) + "m";
    }

    private static void printPullUsage() {
        System.out.println("Usage: jllm pull <owner>/<repo>[/<file.gguf>] [options]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --name <alias>    Register the model under this name (default: filename without .gguf)");
        System.out.println("  --branch <ref>    HuggingFace branch or commit (default: main)");
        System.out.println("  --token <token>   HuggingFace access token for private/gated models");
        System.out.println("                    (can also be set via HF_TOKEN environment variable)");
        System.out.println("  --binary <path>   Path to llama-cli binary (auto-detected if omitted)");
        System.out.println("  --no-register     Download only; do not register in jllm");
        System.out.println("  --no-hash         Skip SHA-256 checksum computation");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  jllm pull bartowski/Llama-3.2-3B-Instruct-GGUF");
        System.out.println("       (lists available GGUF files in the repo)");
        System.out.println("  jllm pull bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
        System.out.println("  jllm pull bartowski/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q4_K_M.gguf --name phi3.5");
        System.out.println("  jllm pull lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --token hf_...");
        System.out.println("  jllm pull bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-F16.gguf --quantize Q4_K_M --name llama3.2:3b");
        System.out.println("       (download F16, quantize to Q4_K_M in-process, register quantized file)");
    }

    private static void cmdVersion() {
        System.out.println("jllm " + Version.JLLM);
        System.out.println();

        System.out.println("Runtime");
        System.out.printf("  Java    : %s (%s)%n",
                System.getProperty("java.version"),
                System.getProperty("java.vendor"));
        System.out.printf("  JVM     : %s %s%n",
                System.getProperty("java.vm.name"),
                System.getProperty("java.vm.version"));
        System.out.printf("  OS      : %s %s (%s)%n",
                System.getProperty("os.name"),
                System.getProperty("os.version"),
                System.getProperty("os.arch"));
        System.out.println();

        System.out.println("JNI");
        boolean jniOk = LlamaModel.isNativeLibraryAvailable();
        System.out.printf("  Status  : %s%n",
                jniOk ? "available" : "not available (subprocess fallback mode)");
        System.out.println();

        System.out.println("Dependencies");
        System.out.printf("  %-16s %s%n", "Gson",          Version.GSON);
        System.out.printf("  %-16s %s%n", "SLF4J",         Version.SLF4J);
        System.out.printf("  %-16s %s%n", "Logback",        Version.LOGBACK);
        System.out.printf("  %-16s %s%n", "Undertow",       Version.UNDERTOW);
        System.out.printf("  %-16s %s%n", "XNIO",           Version.XNIO);
        System.out.printf("  %-16s %s%n", "Apache Lucene",  Version.LUCENE);
        System.out.printf("  %-16s %s%n", "Apache PDFBox",  Version.PDFBOX);
        System.out.println();

        System.out.println("Storage");
        System.out.printf("  %-10s %s%n", "Registry:", ModelRegistry.getRegistryFile());
        System.out.printf("  %-10s %s%n", "Models:",   ModelRegistry.getManagedModelsDir());
        System.out.printf("  %-10s %s%n", "Plugins:",  ModelRegistry.getPluginsDir());
        System.out.printf("  %-10s %s%n", "RAG:",      ModelRegistry.getRagDir());
    }

    // ── rag ───────────────────────────────────────────────────────────────────

    private static void cmdRag(String[] args) throws Exception {
        if (args.length < 2) { printRagUsage(); return; }
        switch (args[1]) {
            case "add":    cmdRagAdd(args);    break;
            case "list":   cmdRagList();       break;
            case "search": cmdRagSearch(args); break;
            case "rm":     cmdRagRm(args);     break;
            default:
                System.err.println("Unknown rag subcommand: " + args[1]);
                printRagUsage();
                System.exit(1);
        }
    }

    private static void cmdRagAdd(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: jllm rag add <collection> <path> [--embed-model <name>]"
                    + " [--chunk-size <words>] [--chunk-overlap <words>]");
            System.exit(1);
        }
        String collection = args[2];
        Path   path       = Paths.get(args[3]);
        RagManager.IndexOptions options = new RagManager.IndexOptions();
        for (int i = 4; i < args.length; i++) {
            switch (args[i]) {
                case "--embed-model":
                    if (i + 1 < args.length) options.embedModel = args[++i];
                    break;
                case "--chunk-size":
                    if (i + 1 < args.length) options.chunkWords = parseIntFlag(args[i], args[++i]);
                    break;
                case "--chunk-overlap":
                    if (i + 1 < args.length) options.overlapWords = parseIntFlag(args[i], args[++i]);
                    break;
                default:
                    System.err.println("Unknown flag: " + args[i]);
                    System.exit(1);
            }
        }
        if (!Files.exists(path)) {
            System.err.println("Path not found: " + path);
            System.exit(1);
        }
        String suffix = (options.embedModel != null ? "  (hybrid: " + options.embedModel + ")" : "")
                + (options.chunkWords != null || options.overlapWords != null
                        ? "  (chunk: " + (options.chunkWords != null ? options.chunkWords : "default")
                          + "/" + (options.overlapWords != null ? options.overlapWords : "default") + ")"
                        : "");
        System.out.println("Indexing into collection '" + collection + "'..." + suffix);
        ragManager.addDocuments(collection, path, options, System.out::println);
        System.out.println("Done.");
        System.out.println("Use: jllm run <model> --rag " + collection);
    }

    private static int parseIntFlag(String flagName, String value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            System.err.println("Invalid value for " + flagName + ": '" + value + "' is not an integer");
            System.exit(1);
            throw new AssertionError("unreachable");
        }
    }

    private static void cmdRagList() throws Exception {
        List<RagManager.CollectionInfo> collections = ragManager.listCollections();
        if (collections.isEmpty()) {
            System.out.println("No RAG collections found.");
            System.out.println("Create one with: jllm rag add <collection> <path>");
            return;
        }
        System.out.printf("%-25s  %8s  %s%n", "COLLECTION", "CHUNKS", "PATH");
        System.out.println("-".repeat(75));
        for (RagManager.CollectionInfo c : collections) {
            String suffix = c.embedModel != null ? "  [hybrid: " + c.embedModel + "]" : "";
            if (c.chunkWords > 0) {
                suffix += "  [chunk:" + c.chunkWords + "/" + c.overlapWords + "]";
            }
            System.out.printf("%-25s  %8d  %s%s%n", c.name, c.chunkCount, c.path, suffix);
        }
    }

    private static void cmdRagSearch(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: jllm rag search <collection> <query>");
            System.exit(1);
        }
        String collection = args[2];
        String query = String.join(" ", java.util.Arrays.copyOfRange(args, 3, args.length));
        System.out.println("Searching '" + collection + "' for: " + query);
        System.out.println();
        List<RagResult> hits = ragManager.search(collection, query);
        if (hits.isEmpty()) {
            System.out.println("No results found.");
            return;
        }
        for (int i = 0; i < hits.size(); i++) {
            RagResult r = hits.get(i);
            String fileName = Paths.get(r.source).getFileName().toString();
            System.out.printf("[%d] %s", i + 1, fileName);
            if (r.page > 0) System.out.printf(" (page %d)", r.page);
            System.out.printf("  score=%.3f", r.score);
            if (!Float.isNaN(r.bm25Score) || !Float.isNaN(r.vectorScore)) {
                System.out.printf("  (bm25=%s, vector=%s)",
                        Float.isNaN(r.bm25Score)   ? "-" : String.format("%.3f", r.bm25Score),
                        Float.isNaN(r.vectorScore) ? "-" : String.format("%.3f", r.vectorScore));
            }
            System.out.println();
            System.out.println("    " + r.content.replaceAll("\\s+", " ")
                    .substring(0, Math.min(200, r.content.length())) + "...");
            System.out.println();
        }
    }

    private static void cmdRagRm(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: jllm rag rm <collection>");
            System.exit(1);
        }
        String collection = args[2];
        ragManager.deleteCollection(collection);
        System.out.println("Deleted collection '" + collection + "'.");
    }

    private static void printRagUsage() {
        System.out.println("Usage: jllm rag <subcommand> [options]");
        System.out.println();
        System.out.println("Subcommands:");
        System.out.println("  add <collection> <path> [--embed-model <name>]");
        System.out.println("                           [--chunk-size <words>] [--chunk-overlap <words>]");
        System.out.println("                                Index a file or directory into a collection.");
        System.out.println("                                --embed-model enables hybrid BM25+vector search");
        System.out.println("                                using the named (already-pulled) GGUF embedding");
        System.out.println("                                model, e.g. nomic-embed-text. Omit for BM25-only");
        System.out.println("                                (default, zero configuration).");
        System.out.println("                                --chunk-size/--chunk-overlap set the chunking");
        System.out.println("                                window in words (default 400/50). Recorded per");
        System.out.println("                                collection and reused on later 'rag add' unless");
        System.out.println("                                overridden.");
        System.out.println("  list                         List all indexed collections");
        System.out.println("  search <collection> <query>  Test retrieval (debug)");
        System.out.println("  rm <collection>              Delete a collection and its index");
        System.out.println();
        System.out.println("Then use a collection in a chat session:");
        System.out.println("  jllm run <model> --rag <collection>");
        System.out.println("Or via API (include in request body):");
        System.out.println("  { \"rag_collection\": \"<collection>\", ... }");
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static void printModelfileParams(ModelConfig m) {
        boolean any = m.getTemperature()  != null || m.getNumPredict()   != null
                   || m.getNumCtx()       != null || m.getNumThreads()   != null
                   || m.getNumGpuLayers() != null
                   || (m.getSystemPrompt() != null && !m.getSystemPrompt().isEmpty());
        if (!any) return;

        System.out.println();
        System.out.println("Parameters:");
        if (m.getTemperature()  != null) System.out.println("  temperature     " + m.getTemperature());
        if (m.getNumPredict()   != null) System.out.println("  num_predict     " + m.getNumPredict());
        if (m.getNumCtx()       != null) System.out.println("  num_ctx         " + m.getNumCtx());
        if (m.getNumThreads()   != null) System.out.println("  num_threads     " + m.getNumThreads());
        if (m.getNumGpuLayers() != null) System.out.println("  num_gpu_layers  " + m.getNumGpuLayers());
        if (m.getSystemPrompt() != null && !m.getSystemPrompt().isEmpty()) {
            String sys = m.getSystemPrompt();
            String preview = sys.length() > 80 ? sys.substring(0, 77) + "..." : sys;
            System.out.println("  system        \"" + preview.replace("\n", "\\n") + "\"");
        }
    }

    /**
     * Detect whether {@code rawPath} is part of a split GGUF and update {@code model}
     * accordingly.  Sets {@code model.path} to the first-shard path, {@code model.shards}
     * to the full ordered list (or {@code null} for single-file), and {@code model.sizeBytes}
     * to the total of all existing shard file sizes.
     */
    private static void applyShardInfo(ModelConfig model, String rawPath) throws Exception {
        Path path  = Paths.get(rawPath);
        SplitGguf.Split split = SplitGguf.detect(path);

        if (split == null || split.totalShards == 1) {
            model.setPath(rawPath);
            model.setShards(null);
            model.setSizeBytes(Files.size(path));
            return;
        }

        // Multi-shard model
        if (!split.isFirstShard()) {
            System.out.printf("  Note: shard %d/%d given; redirecting to shard 1: %s%n",
                    split.shardIndex, split.totalShards, split.first().getFileName());
        }
        System.out.printf("  Split GGUF: %d shards%n", split.totalShards);

        List<Path> missing = split.shards.stream()
                .filter(s -> !Files.exists(s))
                .collect(Collectors.toList());
        if (!missing.isEmpty()) {
            System.out.println("  Warning: " + missing.size() + " shard(s) not found:");
            missing.forEach(s -> System.out.println("    " + s));
        }

        model.setPath(split.first().toString());
        model.setShards(split.shards.stream().map(Path::toString).collect(Collectors.toList()));

        long totalSize = split.existingShards().stream()
                .mapToLong(s -> safeFileSize(s))
                .sum();
        model.setSizeBytes(totalSize);
    }

    /** Returns all shard paths for {@code model}, or a singleton list for single-file models. */
    private static List<Path> getShardPaths(ModelConfig model) {
        if (model.isSplit()) {
            return model.getShards().stream().map(Paths::get).collect(Collectors.toList());
        }
        return Collections.singletonList(Paths.get(model.getPath()));
    }

    private static long safeFileSize(Path p) {
        try { return Files.size(p); } catch (Exception e) { return 0L; }
    }

    /**
     * Read GGUF header metadata from {@code path} and populate the corresponding
     * fields on {@code model}. Silently no-ops on any error (format mismatch,
     * truncated file, etc.) so that registration never fails due to metadata parsing.
     */
    private static void readGgufMetadata(ModelConfig model, String path) {
        try {
            GgufReader.GgufMetadata m = GgufReader.read(Paths.get(path));
            model.setGgufArchitecture(m.architecture);
            model.setGgufQuantization(m.quantization);
            model.setGgufParameterCount(m.parameterCount);
            model.setGgufContextLength(m.contextLength);
            model.setGgufBlockCount(m.blockCount);
            model.setGgufEmbeddingLength(m.embeddingLength);
        } catch (Exception e) {
            System.out.println("  (GGUF metadata: " + e.getMessage() + ")");
        }
    }

    /** Print a one-line summary of extracted GGUF metadata. */
    private static void printGgufSummary(ModelConfig m) {
        boolean any = m.getGgufArchitecture() != null || m.getGgufQuantization() != null
                   || m.getGgufParameterCount() != null || m.getGgufContextLength() != null;
        if (!any) return;
        StringBuilder sb = new StringBuilder("  GGUF:");
        if (m.getGgufArchitecture()   != null) sb.append("  arch=").append(m.getGgufArchitecture());
        if (m.getGgufQuantization()   != null) sb.append("  quant=").append(m.getGgufQuantization());
        if (m.getGgufParameterCount() != null) sb.append("  params=").append(fmtParamCount(m.getGgufParameterCount()));
        if (m.getGgufContextLength()  != null) sb.append("  ctx=").append(m.getGgufContextLength());
        System.out.println(sb);
    }

    private static String fmtParamCount(long p) {
        if (p >= 1_000_000_000_000L) return String.format("%.2fT", p / 1e12);
        if (p >= 1_000_000_000L)     return String.format("%.2fB", p / 1e9);
        if (p >= 1_000_000L)         return String.format("%.1fM", p / 1e6);
        if (p >= 1_000L)             return String.format("%.1fK", p / 1e3);
        return Long.toString(p);
    }

    // ── verify ────────────────────────────────────────────────────────────────

    private static void cmdVerify(String[] args) throws Exception {
        List<ModelConfig> models;
        if (args.length >= 2) {
            ModelConfig m = registry.get(args[1]).orElse(null);
            if (m == null) {
                System.err.println("Model '" + args[1] + "' not found.");
                System.exit(1);
            }
            models = Collections.singletonList(m);
        } else {
            models = registry.list();
            if (models.isEmpty()) { System.out.println("No models registered."); return; }
        }

        System.out.println("Verifying " + models.size() + " model(s)...");
        System.out.println();

        int ok = 0, mismatch = 0, missing = 0, noHash = 0;

        for (ModelConfig m : models) {
            System.out.printf("  %-28s", m.getName());
            List<Path> shardPaths = getShardPaths(m);

            // Check for any missing shard files
            List<Path> missingPaths = shardPaths.stream()
                    .filter(p -> !Files.exists(p))
                    .collect(Collectors.toList());
            if (!missingPaths.isEmpty()) {
                if (m.isSplit()) {
                    System.out.printf("%-12s %d/%d shards missing%n",
                            "MISSING", missingPaths.size(), shardPaths.size());
                } else {
                    System.out.printf("%-12s %s%n", "MISSING", m.getPath());
                }
                missing++;
                continue;
            }
            if (m.getSha256() == null) {
                System.out.printf("%-12s run: jllm update %s --refresh-hash%n", "NO HASH", m.getName());
                noHash++;
                continue;
            }
            String current;
            try {
                current = computeSha256(shardPaths);
            } catch (Exception e) {
                System.out.printf("%-12s %s%n", "ERROR", e.getMessage());
                mismatch++;
                continue;
            }
            if (current.equals(m.getSha256())) {
                String tag = m.isSplit() ? " (" + shardPaths.size() + " shards)" : "";
                System.out.printf("%-12s sha256:%s%s%n", "OK", m.getSha256().substring(0, 16), tag);
                ok++;
            } else {
                System.out.printf("%-12s stored:%s  got:%s%n",
                        "MISMATCH",
                        m.getSha256().substring(0, 16),
                        current.substring(0, 16));
                mismatch++;
            }
        }

        System.out.println();
        System.out.printf("Summary: %d OK", ok);
        if (mismatch > 0) System.out.printf("  %d MISMATCH", mismatch);
        if (missing  > 0) System.out.printf("  %d MISSING",  missing);
        if (noHash   > 0) System.out.printf("  %d NO HASH",  noHash);
        System.out.println();

        if (mismatch > 0 || missing > 0) System.exit(1);
    }

    // ── SHA-256 helpers ───────────────────────────────────────────────────────

    private static final long HASH_PROGRESS_THRESHOLD = 50L * 1024 * 1024; // 50 MB

    /**
     * Compute SHA-256 over a list of files in order (for split GGUFs, this hashes
     * the logical concatenation of all shards). Shows a progress bar when total > 50 MB.
     */
    private static String computeSha256(List<Path> paths) throws Exception {
        long totalSize = paths.stream().mapToLong(p -> safeFileSize(p)).sum();
        boolean prog   = totalSize > HASH_PROGRESS_THRESHOLD;
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] buf   = new byte[64 * 1024];
        long   done  = 0;
        long   startMs = System.currentTimeMillis();

        for (Path path : paths) {
            try (InputStream in = Files.newInputStream(path)) {
                int n;
                while ((n = in.read(buf)) != -1) {
                    md.update(buf, 0, n);
                    done += n;
                    if (prog) printHashProgress(done, totalSize, startMs);
                }
            }
        }
        if (prog) {
            long elapsed = Math.max(1, System.currentTimeMillis() - startMs);
            System.out.printf("\r  SHA-256: computing...  done in %.1f s%s%n",
                    elapsed / 1000.0, " ".repeat(20));
        }

        StringBuilder sb = new StringBuilder(64);
        for (byte b : md.digest()) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    private static String computeSha256(Path path) throws Exception {
        return computeSha256(Collections.singletonList(path));
    }

    private static void printHashProgress(long done, long total, long startMs) {
        long elapsed = Math.max(1, System.currentTimeMillis() - startMs);
        double bps   = done * 1000.0 / elapsed;
        int pct      = (int)(done * 100 / total);
        int width    = 20;
        int filled   = (int)(done * width / total);
        System.out.printf("\r  SHA-256: [%s%s] %3d%%  %s/s          ",
                "█".repeat(filled), "░".repeat(width - filled), pct, formatSize((long) bps));
        System.out.flush();
    }

    /**
     * Compute and store the SHA-256 hash on {@code model}.
     * For split models, hashes all shards in order (logical concatenation).
     * Silently no-ops if {@code noHash} is true.
     */
    private static void computeAndSetSha256(ModelConfig model, boolean noHash) {
        if (noHash) return;
        try {
            model.setSha256(computeSha256(getShardPaths(model)));
        } catch (Exception e) {
            System.out.println("  (SHA-256: " + e.getMessage() + ")");
        }
    }

    /**
     * Warn if any already-registered model shares the same SHA-256 as {@code sha256},
     * excluding the model named {@code exceptName} (the one being added/updated).
     */
    private static void checkForDuplicates(String sha256, String exceptName) {
        if (sha256 == null) return;
        for (ModelConfig m : registry.list()) {
            if (sha256.equals(m.getSha256()) && !exceptName.equals(m.getName())) {
                System.out.printf("  Note: same content as '%s' already registered%n", m.getName());
            }
        }
    }

    /** Print the stored SHA-256 (first 16 hex chars) as a one-liner. */
    private static void printSha256Summary(ModelConfig m) {
        if (m.getSha256() == null) return;
        System.out.println("  SHA-256: " + m.getSha256().substring(0, 16) + "...");
    }

    private static void printUpdateUsage() {
        System.out.println("Usage: jllm update <name> [options]");
        System.out.println();
        System.out.println("  Modify registered model parameters in place.");
        System.out.println("  Only the flags you pass are changed; everything else stays the same.");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --temperature <float>    Set temperature (e.g. 0.7)");
        System.out.println("  --max-tokens <int>       Set max output tokens (num_predict)");
        System.out.println("  --ctx <int>              Set context window size (num_ctx)");
        System.out.println("  --threads <int>          Set CPU thread count (num_threads)");
        System.out.println("  --gpu-layers <int>       Set GPU layers to offload (num_gpu_layers; -1 = all, 0 = CPU only)");
        System.out.println("  --system <text>          Set system prompt");
        System.out.println("  --no-system              Clear system prompt");
        System.out.println("  --path <path>            Update GGUF file path (also recalculates size and refreshes metadata)");
        System.out.println("  --binary <path>          Update llama.cpp binary path");
        System.out.println("  --refresh-gguf           Re-read GGUF metadata from the current file");
        System.out.println("  --refresh-hash           Recompute and store SHA-256 checksum");
        System.out.println("  --unset <param>          Reset a parameter to the runtime default (null)");
        System.out.println("                           Params: temperature | max-tokens | ctx | threads | gpu-layers | system");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  jllm update phi3:mini --temperature 0.2");
        System.out.println("  jllm update phi3:mini --ctx 8192 --threads 8");
        System.out.println("  jllm update phi3:mini --gpu-layers 35");
        System.out.println("  jllm update phi3:mini --system \"You are a coding assistant.\"");
        System.out.println("  jllm update phi3:mini --no-system");
        System.out.println("  jllm update phi3:mini --unset temperature");
        System.out.println("  jllm update phi3:mini --path /new/path/to/model.gguf");
    }

    private static boolean objEq(Object a, Object b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.equals(b);
    }

    private static void printParamDiff(String label, Object before, Object after) {
        if (objEq(before, after)) return;
        System.out.printf("  %-14s %s  →  %s%n", label, paramValStr(before), paramValStr(after));
    }

    private static String paramValStr(Object v) {
        if (v == null) return "(not set)";
        String s = v.toString().replace("\n", "\\n");
        if (s.length() > 55) s = s.substring(0, 52) + "...";
        return s.contains(" ") ? "\"" + s + "\"" : s;
    }

    private static boolean isYamlFile(String path) {
        String lower = path.toLowerCase(Locale.ROOT);
        return lower.endsWith(".yaml") || lower.endsWith(".yml")
            || lower.endsWith("jllmfile");
    }

    private static String detectLlamaBinary() {
        String[] candidates = {
            "/usr/local/bin/llama-cli",
            "/usr/bin/llama-cli",
            System.getProperty("user.home") + "/.local/bin/llama-cli",
            "/opt/llama.cpp/llama-cli",
            "/usr/local/bin/main",
        };
        for (String c : candidates) {
            if (Files.exists(Paths.get(c))) return c;
        }
        return null;
    }

    private static String formatSize(long bytes) {
        if (bytes <= 0) return "-";
        if (bytes < 1024L)               return bytes + " B";
        if (bytes < 1024L * 1024)        return (bytes / 1024) + " KB";
        if (bytes < 1024L * 1024 * 1024) return (bytes / (1024 * 1024)) + " MB";
        return String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private static void printUsage() {
        System.out.println("local-llm — Local LLM manager");
        System.out.println();
        System.out.println("Usage: java -jar local-llm.jar <command> [options]");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  list                                    List registered models with disk status");
        System.out.println("  storage                                 Show per-model disk usage summary");
        System.out.println("  add <name> --path <path>                Register a model by file path");
        System.out.println("             [--binary <path>]            Path to llama.cpp binary");
        System.out.println("             [--format <fmt>]             Model format (default: gguf)");
        System.out.println("             [--managed]                  Copy file into managed storage (~/.local-llm/models/)");
        System.out.println("             [--quantize <type>]          Quantize before registering (requires JNI; e.g. Q4_K_M)");
        System.out.println("             [--quantize-output <path>]   Override output path for quantized file");
        System.out.println("             [--threads <int>]            CPU threads for quantization (default: all)");
        System.out.println("  create <name> -f <file>                 Create a model from a Modelfile or Jllmfile (.yaml/.yml)");
        System.out.println("                [--binary <path>]         Path to llama.cpp binary");
        System.out.println("  rm <name> [--purge]                     Remove a model (--purge also deletes the file)");
        System.out.println("  update <name>                           Update model parameters in place");
        System.out.println("             [--temperature <float>]      Set temperature");
        System.out.println("             [--max-tokens <int>]         Set max output tokens");
        System.out.println("             [--ctx <int>]                Set context window size");
        System.out.println("             [--threads <int>]            Set CPU thread count");
        System.out.println("             [--gpu-layers <int>]         Set GPU layers (-1 = all, 0 = CPU only)");
        System.out.println("             [--system <text>]            Set system prompt");
        System.out.println("             [--no-system]                Clear system prompt");
        System.out.println("             [--path <path>]              Update GGUF file path");
        System.out.println("             [--binary <path>]            Update binary path");
        System.out.println("             [--unset <param>]            Reset a parameter to runtime default");
        System.out.println("  run <name>                              Start an interactive chat session");
        System.out.println("             [--prompt <text>]            One-shot: generate and exit (pipe-friendly)");
        System.out.println("             [--rag <collection>]         Enable RAG retrieval");
        System.out.println("             [--system <prompt>]          Override system prompt for this session");
        System.out.println("             [--no-system]                Clear system prompt for this session");
        System.out.println("             [--temperature <float>]      Override temperature (e.g. 0.2)");
        System.out.println("             [--max-tokens <int>]         Override max output tokens");
        System.out.println("             [--ctx <int>]                Override context window size");
        System.out.println("             [--threads <int>]            Override CPU thread count");
        System.out.println("             [--gpu-layers <int>]         Override GPU layers (-1 = all, 0 = CPU only)");
        System.out.println("             (stdin pipe auto-detected: echo \"...\" | jllm run <name>)");
        System.out.println("  serve [--port <port>]                   Start the HTTP server (default: 11434)");
        System.out.println("        [--max-concurrent <n>]            Max parallel inference slots (default: CPU count)");
        System.out.println("        [--max-body <bytes>]              Max request body size, e.g. 4M (default: 4M)");
        System.out.println("        [--max-tokens <n>]                Server-side output token cap, 0=off (default: 0)");
        System.out.println("        [--rate-limit <req/min>]          Per-IP rate limit, 0=off (default: 0)");
        System.out.println("  ps [--port <port>]                      Show models currently loaded in a running 'jllm serve'");
        System.out.println("  rag add <collection> <path> [--embed-model <name>]");
        System.out.println("           [--chunk-size <words>] [--chunk-overlap <words>]");
        System.out.println("                                          Index a file or directory for RAG");
        System.out.println("  rag list                                List RAG collections");
        System.out.println("  rag search <collection> <query>         Test RAG retrieval");
        System.out.println("  rag rm <collection>                     Delete a RAG collection");
        System.out.println("  show <name> [--yaml]                    Show model config (--yaml for Jllmfile format)");
        System.out.println("  info <name>                             Show model details");
        System.out.println("  pull <owner>/<repo>[/<file.gguf>]       Download a GGUF from HuggingFace and register it");
        System.out.println("       [--name <alias>]                   Register with a custom name");
        System.out.println("       [--branch <ref>]                   HuggingFace branch or commit (default: main)");
        System.out.println("       [--token <token>]                  HF access token (or set HF_TOKEN env var)");
        System.out.println("       [--no-register]                    Download only, skip registration");
        System.out.println("       [--quantize <type>]                Quantize after download before registering (e.g. Q4_K_M)");
        System.out.println("       [--threads <int>]                  CPU threads for quantization (default: all)");
        System.out.println("  verify [<name>]                         Verify SHA-256 checksum(s): OK / MISMATCH / MISSING / NO HASH");
        System.out.println("  plugins                                 List loaded plugin tools and interceptors");
        System.out.println("  plugins reload                          Rescan the plugin directory (see 'jllm run' /reload-plugins");
        System.out.println("                                          and 'jllm serve' /api/plugins/reload for live processes)");
        System.out.println("  version                                 Show jllm version, runtime, and dependency info");
        System.out.println();
        System.out.println("Modelfile example (Ollama-compatible):");
        System.out.println("  FROM /path/to/model.gguf");
        System.out.println("  PARAMETER temperature 0.7");
        System.out.println("  PARAMETER num_predict 1024");
        System.out.println("  PARAMETER num_ctx 4096");
        System.out.println("  SYSTEM You are a helpful assistant.");
        System.out.println();
        System.out.println("Jllmfile example (YAML format, .yaml/.yml/Jllmfile):");
        System.out.println("  from: /path/to/model.gguf");
        System.out.println("  system: \"You are a helpful assistant.\"");
        System.out.println("  parameters:");
        System.out.println("    temperature: 0.7");
        System.out.println("    num_predict: 1024");
        System.out.println("    num_ctx: 4096");
    }
}
