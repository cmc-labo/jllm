package dev.localllm;

import dev.localllm.model.GgufReader;
import dev.localllm.model.JllmfileParser;
import dev.localllm.model.ModelConfig;
import dev.localllm.model.Modelfile;
import dev.localllm.model.ModelRegistry;
import dev.localllm.jni.LlamaModel;
import dev.localllm.plugin.LlmTool;
import dev.localllm.plugin.PluginManager;
import dev.localllm.plugin.PromptInterceptor;
import dev.localllm.pull.HuggingFaceClient;
import dev.localllm.rag.RagManager;
import dev.localllm.rag.RagResult;
import dev.localllm.runner.ModelRunner;
import dev.localllm.server.ApiServer;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class Main {

    private static final ModelRegistry registry   = new ModelRegistry();
    private static final RagManager    ragManager = new RagManager(ModelRegistry.getRagDir());

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
            case "plugins": cmdPlugins();       break;
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
        System.out.printf("%-25s %-10s %-8s %-10s %-9s %s%n", "NAME", "QUANT", "PARAMS", "SIZE", "STATUS", "PATH");
        System.out.println("-".repeat(100));
        long totalBytes = 0;
        int missingCount = 0;
        for (ModelConfig m : models) {
            boolean exists = Files.exists(Paths.get(m.getPath()));
            String status = exists ? "ok" : "missing";
            String quant  = m.getGgufQuantization()   != null ? m.getGgufQuantization()             : "-";
            String params = m.getGgufParameterCount() != null ? fmtParamCount(m.getGgufParameterCount()) : "-";
            System.out.printf("%-25s %-10s %-8s %-10s %-9s %s%n",
                m.getName(), quant, params, formatSize(m.getSizeBytes()), status, m.getPath());
            if (exists) totalBytes += m.getSizeBytes();
            else        missingCount++;
        }
        System.out.println("-".repeat(100));
        System.out.printf("%d model(s)  %s total on disk", models.size(), formatSize(totalBytes));
        if (missingCount > 0) System.out.printf("  (%d file(s) missing — run 'storage' for details)", missingCount);
        System.out.println();
    }

    private static void cmdAdd(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: jllm add <name> --path <path> [--binary <path>] [--format <fmt>] [--managed] [--no-hash]");
            System.exit(1);
        }

        String  name    = args[1];
        String  path    = null;
        String  binary  = null;
        String  format  = "gguf";
        boolean managed = false;
        boolean noHash  = false;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--path":    if (i + 1 < args.length) path   = args[++i]; break;
                case "--binary":  if (i + 1 < args.length) binary = args[++i]; break;
                case "--format":  if (i + 1 < args.length) format = args[++i]; break;
                case "--managed": managed = true; break;
                case "--no-hash": noHash  = true; break;
            }
        }

        if (path == null) { System.err.println("--path is required"); System.exit(1); }
        if (!Files.exists(Paths.get(path))) {
            System.err.println("Model file not found: " + path);
            System.exit(1);
        }

        if (managed) {
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
        model.setPath(path);
        model.setBinary(binary);
        model.setFormat(format);
        model.setSizeBytes(Files.size(Paths.get(path)));
        model.setAddedAt(Instant.now().toString());
        readGgufMetadata(model, path);
        computeAndSetSha256(model, Paths.get(path), noHash);
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

        model.setSizeBytes(Files.size(Paths.get(model.getPath())));
        if (binary != null) {
            model.setBinary(binary);
        } else if (model.getBinary() == null) {
            model.setBinary(detectLlamaBinary());
        }
        readGgufMetadata(model, model.getPath());
        computeAndSetSha256(model, Paths.get(model.getPath()), noHash);
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
        Float   prevTemp    = model.getTemperature();
        Integer prevPredict = model.getNumPredict();
        Integer prevCtx     = model.getNumCtx();
        Integer prevThreads = model.getNumThreads();
        String  prevSystem  = model.getSystemPrompt();
        String  prevPath    = model.getPath();
        String  prevBinary  = model.getBinary();
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
                        model.setPath(p);
                        model.setSizeBytes(Files.size(Paths.get(p)));
                        readGgufMetadata(model, p);
                        computeAndSetSha256(model, Paths.get(p), false);
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
                    computeAndSetSha256(model, Paths.get(model.getPath()), false);
                    ggufRefreshed = true; // reuse the "extra change" flag
                    break;
                case "--unset":
                    if (i + 1 < args.length) {
                        String param = args[++i];
                        switch (param) {
                            case "temperature":                        model.setTemperature(null);   break;
                            case "max-tokens": case "num-predict":     model.setNumPredict(null);    break;
                            case "ctx":        case "num-ctx":         model.setNumCtx(null);        break;
                            case "threads":    case "num-threads":     model.setNumThreads(null);    break;
                            case "system":                             model.setSystemPrompt(null);  break;
                            default:
                                System.err.println("Unknown parameter '" + param + "'.");
                                System.err.println("Unset-able params: temperature  max-tokens  ctx  threads  system");
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
                || !objEq(prevTemp,    model.getTemperature())
                || !objEq(prevPredict, model.getNumPredict())
                || !objEq(prevCtx,     model.getNumCtx())
                || !objEq(prevThreads, model.getNumThreads())
                || !objEq(prevSystem,  model.getSystemPrompt())
                || !objEq(prevPath,    model.getPath())
                || !objEq(prevBinary,  model.getBinary());

        if (!changed) {
            System.out.println("No changes — configuration already matches the given values.");
            return;
        }

        registry.add(model);
        System.out.println("Updated '" + name + "':");
        printParamDiff("temperature", prevTemp,    model.getTemperature());
        printParamDiff("num_predict", prevPredict, model.getNumPredict());
        printParamDiff("num_ctx",     prevCtx,     model.getNumCtx());
        printParamDiff("num_threads", prevThreads, model.getNumThreads());
        printParamDiff("system",      prevSystem,  model.getSystemPrompt());
        printParamDiff("path",        prevPath,    model.getPath());
        printParamDiff("binary",      prevBinary,  model.getBinary());
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
            Path filePath = Paths.get(model.getPath());
            if (Files.exists(filePath)) {
                Files.delete(filePath);
                System.out.println("Deleted file: " + filePath + " (" + formatSize(model.getSizeBytes()) + " freed)");
            } else {
                System.out.println("File not found on disk (already deleted): " + filePath);
            }
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
                                  overrideCtx, overrideThreads, overrideSystem, clearSystem);

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
            String system, boolean clearSystem) {
        ModelConfig c = new ModelConfig();
        c.setName(src.getName());
        c.setPath(src.getPath());
        c.setBinary(src.getBinary());
        c.setFormat(src.getFormat());
        c.setSizeBytes(src.getSizeBytes());
        c.setAddedAt(src.getAddedAt());
        c.setTemperature(temperature  != null ? temperature  : src.getTemperature());
        c.setNumPredict(maxTokens     != null ? maxTokens    : src.getNumPredict());
        c.setNumCtx(ctx               != null ? ctx          : src.getNumCtx());
        c.setNumThreads(threads       != null ? threads      : src.getNumThreads());
        c.setSystemPrompt(clearSystem ? "" : (system != null ? system : src.getSystemPrompt()));
        return c;
    }

    private static void cmdServe(String[] args) throws Exception {
        int port = 11434;
        int maxConcurrent = Runtime.getRuntime().availableProcessors();
        for (int i = 1; i < args.length; i++) {
            if ("--port".equals(args[i]) && i + 1 < args.length) {
                port = Integer.parseInt(args[++i]);
            } else if ("--max-concurrent".equals(args[i]) && i + 1 < args.length) {
                maxConcurrent = Integer.parseInt(args[++i]);
            }
        }
        System.out.println("Starting local-llm server on http://localhost:" + port);
        new ApiServer(port, registry, plugins, ragManager, maxConcurrent).start();
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
        System.out.println("Binary:   " + (m.getBinary()  != null ? m.getBinary()  : "(not set)"));
        System.out.println("Size:     " + formatSize(m.getSizeBytes()));
        System.out.println("Added:    " + (m.getAddedAt() != null ? m.getAddedAt() : "-"));
        System.out.println("SHA-256:  " + (m.getSha256()  != null ? m.getSha256()  : "(not computed — run: jllm update " + m.getName() + " --refresh-hash)"));

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

    /** Show all currently loaded plugins (tools and interceptors). */
    private static void cmdPlugins() {
        Path pluginDir = ModelRegistry.getPluginsDir();
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

        String  ref        = args[1];
        String  name       = null;
        String  branch     = "main";
        String  token      = System.getenv("HF_TOKEN");
        String  binary     = null;
        boolean noRegister = false;
        boolean noHash     = false;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--name":        if (i + 1 < args.length) name   = args[++i]; break;
                case "--branch":      if (i + 1 < args.length) branch = args[++i]; break;
                case "--token":       if (i + 1 < args.length) token  = args[++i]; break;
                case "--binary":      if (i + 1 < args.length) binary = args[++i]; break;
                case "--no-register": noRegister = true; break;
                case "--no-hash":     noHash     = true; break;
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
            if (ggufFiles.size() == 1) {
                filePath = ggufFiles.get(0).name;
                System.out.println("Found: " + filePath);
                System.out.println();
            } else {
                System.out.println("Multiple GGUF files found. Pick one:");
                System.out.println();
                for (HuggingFaceClient.HfFile f : ggufFiles) {
                    System.out.println("  jllm pull " + owner + "/" + repo + "/" + f.name);
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

        // Download (skip if already present).
        if (Files.exists(dest)) {
            System.out.println("Already downloaded: " + dest + "  (" + formatSize(Files.size(dest)) + ")");
            System.out.println("Skipping download. Delete the file first to force a re-download.");
            System.out.println();
        } else {
            System.out.println("Source : " + owner + "/" + repo + "/" + filePath + "  (branch: " + branch + ")");
            System.out.println("Dest   : " + dest);
            System.out.println();

            long[] startMs = {System.currentTimeMillis()};
            try {
                hf.download(owner, repo, filePath, branch, dest, (downloaded, total) ->
                        printPullProgress(downloaded, total, startMs[0]));
            } catch (Exception e) {
                System.out.println();
                System.err.println("Download failed: " + e.getMessage());
                System.exit(1);
            }
            System.out.println(); // end of progress line
            System.out.println("Complete: " + formatSize(Files.size(dest)));
            System.out.println();
        }

        // Register.
        if (!noRegister) {
            if (registry.get(name).isPresent()) {
                System.out.println("Note: '" + name + "' already registered — updating entry.");
            }
            if (binary == null) binary = detectLlamaBinary();
            ModelConfig model = new ModelConfig();
            model.setName(name);
            model.setPath(dest.toString());
            model.setBinary(binary);
            model.setFormat("gguf");
            model.setSizeBytes(Files.size(dest));
            model.setAddedAt(Instant.now().toString());
            readGgufMetadata(model, dest.toString());
            computeAndSetSha256(model, dest, noHash);
            checkForDuplicates(model.getSha256(), name);
            registry.add(model);
            System.out.println("Registered: " + name);
            printGgufSummary(model);
            printSha256Summary(model);
            System.out.println("Run with : jllm run " + name);
        }
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
            System.err.println("Usage: jllm rag add <collection> <path>");
            System.exit(1);
        }
        String collection = args[2];
        Path   path       = Paths.get(args[3]);
        if (!Files.exists(path)) {
            System.err.println("Path not found: " + path);
            System.exit(1);
        }
        System.out.println("Indexing into collection '" + collection + "'...");
        ragManager.addDocuments(collection, path, System.out::println);
        System.out.println("Done.");
        System.out.println("Use: jllm run <model> --rag " + collection);
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
            System.out.printf("%-25s  %8d  %s%n", c.name, c.chunkCount, c.path);
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
            System.out.printf("  score=%.3f%n", r.score);
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
        System.out.println("  add <collection> <path>      Index a file or directory into a collection");
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
        boolean any = m.getTemperature() != null || m.getNumPredict() != null
                   || m.getNumCtx()     != null || m.getNumThreads() != null
                   || (m.getSystemPrompt() != null && !m.getSystemPrompt().isEmpty());
        if (!any) return;

        System.out.println();
        System.out.println("Parameters:");
        if (m.getTemperature()  != null) System.out.println("  temperature   " + m.getTemperature());
        if (m.getNumPredict()   != null) System.out.println("  num_predict   " + m.getNumPredict());
        if (m.getNumCtx()       != null) System.out.println("  num_ctx       " + m.getNumCtx());
        if (m.getNumThreads()   != null) System.out.println("  num_threads   " + m.getNumThreads());
        if (m.getSystemPrompt() != null && !m.getSystemPrompt().isEmpty()) {
            String sys = m.getSystemPrompt();
            String preview = sys.length() > 80 ? sys.substring(0, 77) + "..." : sys;
            System.out.println("  system        \"" + preview.replace("\n", "\\n") + "\"");
        }
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
            Path filePath = Paths.get(m.getPath());

            if (!Files.exists(filePath)) {
                System.out.printf("%-12s %s%n", "MISSING", m.getPath());
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
                current = computeSha256(filePath);
            } catch (Exception e) {
                System.out.printf("%-12s %s%n", "ERROR", e.getMessage());
                mismatch++;
                continue;
            }
            if (current.equals(m.getSha256())) {
                System.out.printf("%-12s sha256:%s%n", "OK", m.getSha256().substring(0, 16));
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
     * Compute the SHA-256 hex digest of a file.
     * Shows an inline progress bar for files larger than {@link #HASH_PROGRESS_THRESHOLD}.
     */
    private static String computeSha256(Path path) throws Exception {
        long size    = Files.size(path);
        boolean prog = size > HASH_PROGRESS_THRESHOLD;
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] buf   = new byte[64 * 1024];
        long   done  = 0;
        long   startMs = System.currentTimeMillis();

        try (InputStream in = Files.newInputStream(path)) {
            int n;
            while ((n = in.read(buf)) != -1) {
                md.update(buf, 0, n);
                done += n;
                if (prog) printHashProgress(done, size, startMs);
            }
        }
        if (prog) {
            // Overwrite progress line with completion
            long elapsed = Math.max(1, System.currentTimeMillis() - startMs);
            System.out.printf("\r  SHA-256: computing...  done in %.1f s%s%n",
                    elapsed / 1000.0, " ".repeat(20));
        }

        StringBuilder sb = new StringBuilder(64);
        for (byte b : md.digest()) sb.append(String.format("%02x", b));
        return sb.toString();
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
     * Silently no-ops if {@code noHash} is true.
     */
    private static void computeAndSetSha256(ModelConfig model, Path path, boolean noHash) {
        if (noHash) return;
        try {
            model.setSha256(computeSha256(path));
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
        System.out.println("  --system <text>          Set system prompt");
        System.out.println("  --no-system              Clear system prompt");
        System.out.println("  --path <path>            Update GGUF file path (also recalculates size and refreshes metadata)");
        System.out.println("  --binary <path>          Update llama.cpp binary path");
        System.out.println("  --refresh-gguf           Re-read GGUF metadata from the current file");
        System.out.println("  --refresh-hash           Recompute and store SHA-256 checksum");
        System.out.println("  --unset <param>          Reset a parameter to the runtime default (null)");
        System.out.println("                           Params: temperature | max-tokens | ctx | threads | system");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  jllm update phi3:mini --temperature 0.2");
        System.out.println("  jllm update phi3:mini --ctx 8192 --threads 8");
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
        System.out.println("  create <name> -f <file>                 Create a model from a Modelfile or Jllmfile (.yaml/.yml)");
        System.out.println("                [--binary <path>]         Path to llama.cpp binary");
        System.out.println("  rm <name> [--purge]                     Remove a model (--purge also deletes the file)");
        System.out.println("  update <name>                           Update model parameters in place");
        System.out.println("             [--temperature <float>]      Set temperature");
        System.out.println("             [--max-tokens <int>]         Set max output tokens");
        System.out.println("             [--ctx <int>]                Set context window size");
        System.out.println("             [--threads <int>]            Set CPU thread count");
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
        System.out.println("             (stdin pipe auto-detected: echo \"...\" | jllm run <name>)");
        System.out.println("  serve [--port <port>]                   Start the HTTP server (default: 11434)");
        System.out.println("        [--max-concurrent <n>]            Max parallel inference slots (default: CPU count)");
        System.out.println("  rag add <collection> <path>             Index a file or directory for RAG");
        System.out.println("  rag list                                List RAG collections");
        System.out.println("  rag search <collection> <query>         Test RAG retrieval");
        System.out.println("  rag rm <collection>                     Delete a RAG collection");
        System.out.println("  show <name> [--yaml]                    Show model config (--yaml for Jllmfile format)");
        System.out.println("  info <name>                             Show model details");
        System.out.println("  pull <owner>/<repo>[/<file.gguf>]        Download a GGUF from HuggingFace and register it");
        System.out.println("       [--name <alias>]                   Register with a custom name");
        System.out.println("       [--branch <ref>]                   HuggingFace branch or commit (default: main)");
        System.out.println("       [--token <token>]                  HF access token (or set HF_TOKEN env var)");
        System.out.println("       [--no-register]                    Download only, skip registration");
        System.out.println("  verify [<name>]                         Verify SHA-256 checksum(s): OK / MISMATCH / MISSING / NO HASH");
        System.out.println("  plugins                                 List loaded plugin tools and interceptors");
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
