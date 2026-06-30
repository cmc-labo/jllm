package dev.localllm.runner;

import ch.qos.logback.classic.Level;
import dev.localllm.jni.LlamaContext;
import dev.localllm.jni.LlamaModel;
import dev.localllm.model.ModelConfig;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs an interactive chat session for a registered model.
 *
 * <p>Two execution paths are supported:
 * <ol>
 *   <li><b>JNI (default)</b> — loads the GGUF file in-process via
 *       {@link LlamaModel} and streams tokens to the terminal as they are
 *       produced by {@link LlamaContext#generateStreaming}. No external
 *       binary is required.</li>
 *   <li><b>Subprocess (fallback)</b> — shells out to a {@code llama-cli}
 *       binary when the JNI native library is not available. Used only if
 *       {@code --binary} was supplied at registration time.</li>
 * </ol>
 */
public class ModelRunner {

    private static final int   DEFAULT_N_CTX      = 4096;
    private static final int   DEFAULT_N_THREADS  = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final float DEFAULT_TEMPERATURE = 0.8f;
    private static final int   DEFAULT_NUM_PREDICT = 512;

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Starts an interactive chat session. Tries the JNI path first; falls
     * back to the subprocess path if the native library is unavailable.
     */
    public void runInteractive(ModelConfig model) throws Exception {
        try {
            runInteractiveJni(model);
        } catch (UnsatisfiedLinkError e) {
            if (model.getBinary() == null) {
                throw new RuntimeException(
                    "JNI native library not found and no llama-cli binary configured for '"
                    + model.getName() + "'.\n"
                    + "Build the native library (see README) or re-register with --binary <path>.");
            }
            System.out.println("Note: JNI native library not available — falling back to llama-cli subprocess.");
            System.out.println();
            runInteractiveSubprocess(model);
        }
    }

    /** Used by ApiServer for single-shot generation via subprocess. */
    public String generate(ModelConfig model, String prompt, int maxTokens) throws Exception {
        if (model.getBinary() == null) {
            throw new RuntimeException("No binary configured for '" + model.getName() + "'.");
        }

        List<String> cmd = new ArrayList<>();
        cmd.add(model.getBinary());
        cmd.add("-m"); cmd.add(model.getPath());
        cmd.add("-p"); cmd.add(prompt);
        cmd.add("-n"); cmd.add(String.valueOf(maxTokens));
        cmd.add("--no-display-prompt");

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(false);
        Process process = pb.start();

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) output.append(line).append("\n");
        }
        int exitCode = process.waitFor();
        if (exitCode != 0) throw new RuntimeException("llama process exited with code: " + exitCode);
        return output.toString().trim();
    }

    // ── JNI interactive loop ──────────────────────────────────────────────────

    private void runInteractiveJni(ModelConfig model) throws Exception {
        int   nCtx        = model.getNumCtx()      != null ? model.getNumCtx()      : DEFAULT_N_CTX;
        int   nThreads    = model.getNumThreads()  != null ? model.getNumThreads()  : DEFAULT_N_THREADS;
        float temperature = model.getTemperature() != null ? model.getTemperature() : DEFAULT_TEMPERATURE;
        int   numPredict  = model.getNumPredict()  != null ? model.getNumPredict()  : DEFAULT_NUM_PREDICT;
        String system     = model.getSystemPrompt();

        printHeader(model.getName(), system, temperature, numPredict, nCtx);

        // history: alternating [role, content] pairs
        List<String[]> history = new ArrayList<>();

        try (LlamaModel llama = new LlamaModel(model.getPath(), 0);
             BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in))) {

            // Create context once; reuse across turns (llama.cpp re-encodes the
            // full prompt each call, overwriting the KV cache from position 0).
            LlamaContext ctx = llama.createContext(nCtx, nThreads);

            // Suppress verbose native INFO logs now that model + context are ready.
            quietNativeLogs();

            try {
                while (true) {
                    String input = readUserInput(stdin);
                    if (input == null) break; // EOF (Ctrl+D)

                    // ── slash commands ─────────────────────────────────────
                    if ("/quit".equals(input) || "/exit".equals(input) || "/bye".equals(input)) {
                        System.out.println("Goodbye!");
                        break;
                    }
                    if ("/clear".equals(input)) {
                        history.clear();
                        ctx.close();
                        ctx = llama.createContext(nCtx, nThreads);
                        System.out.println("[History cleared]");
                        continue;
                    }
                    if ("/help".equals(input)) {
                        printHelp();
                        continue;
                    }
                    if (input.isEmpty()) continue;

                    // ── generate ───────────────────────────────────────────
                    history.add(new String[]{"user", input});
                    String prompt = buildPrompt(history, system);

                    System.out.print("\nAssistant> ");
                    System.out.flush();

                    StringBuilder response = new StringBuilder();
                    ctx.generateStreaming(prompt, numPredict, temperature, piece -> {
                        System.out.print(piece);
                        System.out.flush();
                        response.append(piece);
                    });
                    System.out.println();

                    String assistantText = response.toString().trim();
                    if (!assistantText.isEmpty()) {
                        history.add(new String[]{"assistant", assistantText});
                    }
                }
            } finally {
                ctx.close();
            }
        }
    }

    // ── Subprocess fallback ───────────────────────────────────────────────────

    private void runInteractiveSubprocess(ModelConfig model) throws Exception {
        List<String> cmd = new ArrayList<>();
        cmd.add(model.getBinary());
        cmd.add("-m"); cmd.add(model.getPath());
        cmd.add("-i");
        cmd.add("--chat-template"); cmd.add("chatml");
        cmd.add("-c"); cmd.add(model.getNumCtx() != null ? String.valueOf(model.getNumCtx()) : "4096");
        if (model.getNumThreads()  != null) { cmd.add("-t");       cmd.add(String.valueOf(model.getNumThreads())); }
        if (model.getTemperature() != null) { cmd.add("--temp");   cmd.add(String.valueOf(model.getTemperature())); }
        if (model.getSystemPrompt() != null && !model.getSystemPrompt().isEmpty()) {
            cmd.add("--system"); cmd.add(model.getSystemPrompt());
        }

        System.out.println("Starting " + model.getName() + "  (Ctrl+C to quit)");
        System.out.println();

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        int exitCode = pb.start().waitFor();
        if (exitCode != 0) System.err.println("Model exited with code: " + exitCode);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Builds a ChatML-formatted prompt from the full conversation history.
     * A new LlamaContext is created per turn so the entire history is
     * re-processed on each call; this keeps context management simple at the
     * cost of O(n²) total decode work for long conversations.
     */
    private static String buildPrompt(List<String[]> history, String system) {
        StringBuilder sb = new StringBuilder();
        if (system != null && !system.isEmpty()) {
            sb.append("<|im_start|>system\n").append(system).append("<|im_end|>\n");
        }
        for (String[] msg : history) {
            sb.append("<|im_start|>").append(msg[0]).append("\n")
              .append(msg[1]).append("<|im_end|>\n");
        }
        return sb.append("<|im_start|>assistant\n").toString();
    }

    private static String readUserInput(BufferedReader stdin) throws Exception {
        System.out.print("\nYou> ");
        System.out.flush();
        String line = stdin.readLine();
        if (line == null) return null;
        return line.trim();
    }

    private static void printHeader(String name, String system, float temp, int numPredict, int nCtx) {
        System.out.println("Model    : " + name);
        if (system != null && !system.isEmpty()) {
            String preview = system.replace("\n", " ");
            if (preview.length() > 72) preview = preview.substring(0, 69) + "...";
            System.out.println("System   : " + preview);
        }
        System.out.printf("Settings : temperature=%.2f  max_tokens=%d  context=%d%n", temp, numPredict, nCtx);
        System.out.println("Commands : /clear  /help  /quit");
        System.out.println("-".repeat(60));
    }

    /** Drops native library INFO chatter once the model and context are loaded. */
    private static void quietNativeLogs() {
        try {
            ch.qos.logback.classic.Logger nativeLogger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("dev.localllm.native");
            nativeLogger.setLevel(Level.WARN);
        } catch (ClassCastException ignored) {
            // Non-logback binding in use; skip silently.
        }
    }

    private static void printHelp() {
        System.out.println();
        System.out.println("  /clear   Clear conversation history and start fresh");
        System.out.println("  /help    Show this message");
        System.out.println("  /quit    Exit the chat  (also: /exit, /bye, Ctrl+D)");
        System.out.println();
    }
}
