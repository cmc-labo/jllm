package dev.localllm;

import dev.localllm.model.JllmfileParser;
import dev.localllm.model.ModelConfig;
import dev.localllm.model.Modelfile;
import dev.localllm.model.ModelRegistry;
import dev.localllm.plugin.LlmTool;
import dev.localllm.plugin.PluginManager;
import dev.localllm.plugin.PromptInterceptor;
import dev.localllm.runner.ModelRunner;
import dev.localllm.server.ApiServer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.List;
import java.util.Locale;

public class Main {

    private static final ModelRegistry registry = new ModelRegistry();

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
            case "show":    cmdShow(args);      break;
            case "info":    cmdInfo(args);      break;
            case "storage": cmdStorage();       break;
            case "plugins": cmdPlugins();       break;
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
        System.out.printf("%-25s %-8s %-10s %-9s %s%n", "NAME", "FORMAT", "SIZE", "STATUS", "PATH");
        System.out.println("-".repeat(90));
        long totalBytes = 0;
        int missingCount = 0;
        for (ModelConfig m : models) {
            boolean exists = Files.exists(Paths.get(m.getPath()));
            String status = exists ? "ok" : "missing";
            System.out.printf("%-25s %-8s %-10s %-9s %s%n",
                m.getName(),
                m.getFormat() != null ? m.getFormat() : "-",
                formatSize(m.getSizeBytes()),
                status,
                m.getPath());
            if (exists) totalBytes += m.getSizeBytes();
            else        missingCount++;
        }
        System.out.println("-".repeat(90));
        System.out.printf("%d model(s)  %s total on disk", models.size(), formatSize(totalBytes));
        if (missingCount > 0) System.out.printf("  (%d file(s) missing — run 'storage' for details)", missingCount);
        System.out.println();
    }

    private static void cmdAdd(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: jllm add <name> --path <path> [--binary <path>] [--format <fmt>] [--managed]");
            System.exit(1);
        }

        String name    = args[1];
        String path    = null;
        String binary  = null;
        String format  = "gguf";
        boolean managed = false;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--path":    if (i + 1 < args.length) path   = args[++i]; break;
                case "--binary":  if (i + 1 < args.length) binary = args[++i]; break;
                case "--format":  if (i + 1 < args.length) format = args[++i]; break;
                case "--managed": managed = true; break;
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

        registry.add(model);
        System.out.println("Registered '" + name + "' (" + formatSize(model.getSizeBytes()) + ")");
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

        String name = args[1];
        String modelfilePath = null;
        String binary = null;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "-f":
                case "--file":   if (i + 1 < args.length) modelfilePath = args[++i]; break;
                case "--binary": if (i + 1 < args.length) binary        = args[++i]; break;
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

        registry.add(model);
        System.out.println("Created '" + name + "' (" + formatSize(model.getSizeBytes()) + ")");
        printModelfileParams(model);
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
        if (args.length < 2) { System.err.println("Usage: local-llm run <name>"); System.exit(1); }
        ModelConfig model = registry.get(args[1]).orElse(null);
        if (model == null) {
            System.err.println("Model '" + args[1] + "' not found. Run 'list' to see available models.");
            System.exit(1);
        }
        new ModelRunner(plugins).runInteractive(model);
    }

    private static void cmdServe(String[] args) throws Exception {
        int port = 11434;
        for (int i = 1; i < args.length; i++) {
            if ("--port".equals(args[i]) && i + 1 < args.length) {
                port = Integer.parseInt(args[++i]);
            }
        }
        System.out.println("Starting local-llm server on http://localhost:" + port);
        new ApiServer(port, registry, plugins).start();
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
        System.out.println("  run <name>                              Start an interactive chat session");
        System.out.println("  serve [--port <port>]                   Start the HTTP server (default: 11434)");
        System.out.println("  show <name> [--yaml]                    Show model config (--yaml for Jllmfile format)");
        System.out.println("  info <name>                             Show model details");
        System.out.println("  plugins                                 List loaded plugin tools and interceptors");
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
