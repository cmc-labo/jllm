package dev.localllm;

import dev.localllm.model.ModelConfig;
import dev.localllm.model.ModelRegistry;
import dev.localllm.runner.ModelRunner;
import dev.localllm.server.ApiServer;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.List;

public class Main {

    private static final ModelRegistry registry = new ModelRegistry();

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            printUsage();
            return;
        }

        String cmd = args[0];
        if ("list".equals(cmd)) {
            cmdList();
        } else if ("add".equals(cmd)) {
            cmdAdd(args);
        } else if ("rm".equals(cmd) || "remove".equals(cmd)) {
            cmdRemove(args);
        } else if ("run".equals(cmd)) {
            cmdRun(args);
        } else if ("serve".equals(cmd)) {
            cmdServe(args);
        } else if ("info".equals(cmd)) {
            cmdInfo(args);
        } else {
            System.err.println("Unknown command: " + cmd);
            printUsage();
            System.exit(1);
        }
    }

    // ── commands ──────────────────────────────────────────────────────────────

    private static void cmdList() {
        List<ModelConfig> models = registry.list();
        if (models.isEmpty()) {
            System.out.println("No models registered. Use 'add' to register one.");
            return;
        }
        System.out.printf("%-25s %-8s %-10s %s%n", "NAME", "FORMAT", "SIZE", "PATH");
        System.out.println("-".repeat(80));
        for (ModelConfig m : models) {
            System.out.printf("%-25s %-8s %-10s %s%n",
                m.getName(),
                m.getFormat() != null ? m.getFormat() : "-",
                formatSize(m.getSizeBytes()),
                m.getPath());
        }
    }

    private static void cmdAdd(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: local-llm add <name> --path <path> [--binary <binary>] [--format <fmt>]");
            System.exit(1);
        }

        String name   = args[1];
        String path   = null;
        String binary = null;
        String format = "gguf";

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--path":
                    if (i + 1 < args.length) path = args[++i];
                    break;
                case "--binary":
                    if (i + 1 < args.length) binary = args[++i];
                    break;
                case "--format":
                    if (i + 1 < args.length) format = args[++i];
                    break;
            }
        }

        if (path == null) {
            System.err.println("--path is required");
            System.exit(1);
        }
        if (!Files.exists(Paths.get(path))) {
            System.err.println("Model file not found: " + path);
            System.exit(1);
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
        if (binary != null) {
            System.out.println("Binary: " + binary);
        } else {
            System.out.println("Warning: llama.cpp binary not found. Specify with --binary.");
        }
    }

    private static void cmdRemove(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: local-llm rm <name>");
            System.exit(1);
        }
        String name = args[1];
        if (registry.remove(name)) {
            System.out.println("Removed '" + name + "'.");
        } else {
            System.err.println("Model '" + name + "' not found.");
            System.exit(1);
        }
    }

    private static void cmdRun(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: local-llm run <name>");
            System.exit(1);
        }
        ModelConfig model = registry.get(args[1]).orElse(null);
        if (model == null) {
            System.err.println("Model '" + args[1] + "' not found. Run 'list' to see available models.");
            System.exit(1);
        }
        new ModelRunner().runInteractive(model);
    }

    private static void cmdServe(String[] args) throws Exception {
        int port = 11434;
        for (int i = 1; i < args.length; i++) {
            if ("--port".equals(args[i]) && i + 1 < args.length) {
                port = Integer.parseInt(args[++i]);
            }
        }
        System.out.println("Starting local-llm server on http://localhost:" + port);
        new ApiServer(port, registry).start();
    }

    private static void cmdInfo(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: local-llm info <name>");
            System.exit(1);
        }
        ModelConfig m = registry.get(args[1]).orElse(null);
        if (m == null) {
            System.err.println("Model '" + args[1] + "' not found.");
            System.exit(1);
        }
        System.out.println("Name:    " + m.getName());
        System.out.println("Format:  " + (m.getFormat() != null ? m.getFormat() : "-"));
        System.out.println("Path:    " + m.getPath());
        System.out.println("Binary:  " + (m.getBinary() != null ? m.getBinary() : "(not set)"));
        System.out.println("Size:    " + formatSize(m.getSizeBytes()));
        System.out.println("Added:   " + (m.getAddedAt() != null ? m.getAddedAt() : "-"));
    }

    // ── helpers ───────────────────────────────────────────────────────────────

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
        if (bytes < 1024L)              return bytes + " B";
        if (bytes < 1024L * 1024)       return (bytes / 1024) + " KB";
        if (bytes < 1024L * 1024 * 1024) return (bytes / (1024 * 1024)) + " MB";
        return String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private static void printUsage() {
        System.out.println("local-llm — Local LLM manager");
        System.out.println();
        System.out.println("Usage: java -jar local-llm.jar <command> [options]");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  list                              登録済みモデルの一覧");
        System.out.println("  add <name> --path <path>          モデルを登録");
        System.out.println("             [--binary <binary>]    llama.cppバイナリのパス");
        System.out.println("             [--format <format>]    モデル形式 (default: gguf)");
        System.out.println("  rm <name>                         モデルを削除");
        System.out.println("  run <name>                        インタラクティブチャット");
        System.out.println("  serve [--port <port>]             HTTPサーバー起動 (default: 11434)");
        System.out.println("  info <name>                       モデル詳細の表示");
    }
}
