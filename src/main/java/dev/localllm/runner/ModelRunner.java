package dev.localllm.runner;

import ch.qos.logback.classic.Level;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import dev.localllm.jni.LlamaContext;
import dev.localllm.jni.LlamaModel;
import dev.localllm.model.ModelConfig;
import dev.localllm.plugin.LlmTool;
import dev.localllm.plugin.PluginManager;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Runs an interactive chat session for a registered model.
 *
 * <p>Two execution paths are supported:
 * <ol>
 *   <li><b>JNI (default)</b> — loads the GGUF file in-process via
 *       {@link LlamaModel} and streams tokens to the terminal as they arrive.
 *       No external binary is required.</li>
 *   <li><b>Subprocess (fallback)</b> — shells out to a {@code llama-cli}
 *       binary when the JNI native library is not available.</li>
 * </ol>
 *
 * <p>When a {@link PluginManager} is supplied:
 * <ul>
 *   <li>{@link dev.localllm.plugin.PromptInterceptor}s are applied to every
 *       assembled ChatML prompt before it reaches the model.</li>
 *   <li>{@link LlmTool}s are advertised in the system prompt; the model may
 *       reply with {@code <tool_call>{...}</tool_call>} and jllm will execute
 *       the tool and inject the result back into the conversation (up to
 *       {@value #MAX_TOOL_TURNS} times per user message).</li>
 * </ul>
 */
public class ModelRunner {

    private static final int   DEFAULT_N_CTX       = 4096;
    private static final int   DEFAULT_N_THREADS   = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final float DEFAULT_TEMPERATURE  = 0.8f;
    private static final int   DEFAULT_NUM_PREDICT  = 512;

    /** Maximum consecutive tool calls before returning control to the user. */
    private static final int MAX_TOOL_TURNS = 5;

    private static final Pattern TOOL_CALL_RE =
            Pattern.compile("<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", Pattern.DOTALL);

    private static final Gson GSON = new Gson();

    private final PluginManager plugins;

    public ModelRunner() {
        this(PluginManager.EMPTY);
    }

    public ModelRunner(PluginManager plugins) {
        this.plugins = plugins != null ? plugins : PluginManager.EMPTY;
    }

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
        String baseSystem = model.getSystemPrompt();

        // If tools are loaded, append tool-use instructions to the system prompt.
        String effectiveSystem = plugins.hasTools()
                ? buildSystemWithTools(baseSystem, plugins.getTools())
                : baseSystem;

        printHeader(model.getName(), baseSystem, temperature, numPredict, nCtx);
        if (plugins.hasTools()) {
            System.out.printf("Tools    : %d loaded (%s)%n",
                    plugins.getTools().size(),
                    plugins.getTools().stream().map(LlmTool::getName)
                           .reduce((a, b) -> a + ", " + b).orElse(""));
        }
        if (plugins.hasInterceptors()) {
            System.out.printf("Intercept: %d loaded%n", plugins.getInterceptors().size());
        }

        List<String[]> history = new ArrayList<>(); // [role, content] pairs

        try (LlamaModel llama = new LlamaModel(model.getPath(), 0);
             BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in))) {

            // Create context once; reuse across turns.
            LlamaContext ctx = llama.createContext(nCtx, nThreads);
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
                        printHelp(plugins);
                        continue;
                    }
                    if (input.isEmpty()) continue;

                    // ── inner tool-call loop ───────────────────────────────
                    history.add(new String[]{"user", input});

                    for (int toolTurn = 0; toolTurn <= MAX_TOOL_TURNS; toolTurn++) {
                        if (toolTurn == MAX_TOOL_TURNS) {
                            System.out.println("[Reached max tool calls (" + MAX_TOOL_TURNS + ") — returning to prompt]");
                            break;
                        }

                        String prompt = buildPrompt(history, effectiveSystem);
                        prompt = plugins.applyInterceptors(prompt);

                        System.out.print("\nAssistant> ");
                        System.out.flush();

                        StringBuilder response = new StringBuilder();
                        ctx.generateStreaming(prompt, numPredict, temperature, piece -> {
                            System.out.print(piece);
                            System.out.flush();
                            response.append(piece);
                        });
                        System.out.println();

                        String text = response.toString().trim();
                        history.add(new String[]{"assistant", text});

                        // Detect tool call — only when tools are actually loaded.
                        if (!plugins.hasTools()) break;
                        String callJson = extractToolCall(text);
                        if (callJson == null) break;

                        String result = runToolCall(callJson, plugins.getTools());
                        System.out.println("[Tool result] " + result);
                        history.add(new String[]{"user", "[Tool result] " + result});
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
        if (model.getNumThreads()  != null) { cmd.add("-t");     cmd.add(String.valueOf(model.getNumThreads())); }
        if (model.getTemperature() != null) { cmd.add("--temp"); cmd.add(String.valueOf(model.getTemperature())); }
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

    // ── Prompt builders ───────────────────────────────────────────────────────

    /**
     * Builds a ChatML-formatted prompt from the full conversation history.
     * The entire history is re-processed on each call (new decode from position 0)
     * which keeps context management simple at the cost of O(n²) decode work
     * for long conversations.
     */
    static String buildPrompt(List<String[]> history, String system) {
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

    /**
     * Appends tool-use instructions and tool descriptions to the system prompt.
     * Returns the base system prompt unchanged if no tools are provided.
     */
    static String buildSystemWithTools(String baseSystem, List<LlmTool> tools) {
        if (tools.isEmpty()) return baseSystem;

        StringBuilder sb = new StringBuilder();
        if (baseSystem != null && !baseSystem.isEmpty()) {
            sb.append(baseSystem).append("\n\n");
        }
        sb.append("You have access to the following tools. To invoke a tool, reply with ONLY:\n");
        sb.append("<tool_call>{\"name\":\"TOOL_NAME\",\"args\":{...}}</tool_call>\n");
        sb.append("Do not include any other text when invoking a tool.\n\n");
        sb.append("Available tools:\n");
        for (LlmTool tool : tools) {
            sb.append("- ").append(tool.getName()).append(": ").append(tool.getDescription()).append("\n");
            sb.append("  Parameters: ").append(tool.getParametersSchema()).append("\n");
        }
        return sb.toString().stripTrailing();
    }

    // ── Tool call execution ───────────────────────────────────────────────────

    /** Returns the JSON payload inside {@code <tool_call>...</tool_call>}, or {@code null}. */
    static String extractToolCall(String text) {
        Matcher m = TOOL_CALL_RE.matcher(text);
        return m.find() ? m.group(1).trim() : null;
    }

    /** Parse a tool call JSON, dispatch to the matching tool, and return the result string. */
    static String runToolCall(String callJson, List<LlmTool> tools) {
        String toolName = null;
        String argsJson = "{}";
        try {
            JsonObject call = GSON.fromJson(callJson, JsonObject.class);
            toolName = call.get("name").getAsString();
            JsonElement args = call.get("args");
            if (args != null && args.isJsonObject()) argsJson = args.toString();
        } catch (Exception e) {
            return "[Error: could not parse tool_call JSON: " + e.getMessage() + "]";
        }

        final String name = toolName;
        LlmTool tool = tools.stream().filter(t -> t.getName().equals(name)).findFirst().orElse(null);
        if (tool == null) {
            return "[Error: no tool named '" + name + "'. Available: "
                    + tools.stream().map(LlmTool::getName).reduce((a, b) -> a + ", " + b).orElse("none") + "]";
        }

        try {
            return tool.execute(argsJson);
        } catch (Exception e) {
            return "[Error executing '" + name + "': " + e.getMessage() + "]";
        }
    }

    // ── Terminal helpers ──────────────────────────────────────────────────────

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

    private static void printHelp(PluginManager pm) {
        System.out.println();
        System.out.println("  /clear   Clear conversation history and start fresh");
        System.out.println("  /help    Show this message");
        System.out.println("  /quit    Exit the chat  (also: /exit, /bye, Ctrl+D)");
        if (pm.hasTools()) {
            System.out.println();
            System.out.println("  Loaded tools:");
            for (LlmTool t : pm.getTools()) {
                System.out.printf("    %-18s %s%n", t.getName(), t.getDescription());
            }
        }
        System.out.println();
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
}
