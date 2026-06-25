package dev.localllm.model;

import java.util.Locale;

/**
 * Minimal Ollama-compatible Modelfile parser and serializer.
 *
 * <p>Supported instructions:
 * <pre>
 *   FROM &lt;path&gt;                        Path to a GGUF model file.
 *   PARAMETER temperature 0.8           Sampling temperature (float).
 *   PARAMETER num_predict 512           Max tokens to generate (int).
 *   PARAMETER num_ctx     4096          Context window size (int).
 *   PARAMETER num_threads 4             CPU thread count (int).
 *   SYSTEM &lt;text&gt;                       Single-line system prompt.
 *   SYSTEM """                          Multi-line system prompt.
 *   ...content...
 *   """
 *   # comment                           Ignored.
 * </pre>
 *
 * Unknown instructions are silently ignored so Modelfiles written for full
 * Ollama (ADAPTER, TEMPLATE, MESSAGE, ...) can still be used here without
 * parse errors; the unsupported fields simply have no effect.
 */
public final class Modelfile {

    private Modelfile() {}

    /**
     * Parse {@code content} and apply the recognised instructions to
     * {@code config}. Existing fields that a Modelfile instruction does not
     * cover are left unchanged.
     */
    public static void apply(String content, ModelConfig config) {
        String[] lines = content.split("\\r?\\n", -1);
        StringBuilder systemAccum = null; // non-null while inside """ block

        for (int i = 0; i < lines.length; i++) {
            // Inside a multi-line SYSTEM block
            if (systemAccum != null) {
                String trimmed = lines[i].strip();
                if (trimmed.equals("\"\"\"")) {
                    config.setSystemPrompt(systemAccum.toString().stripTrailing());
                    systemAccum = null;
                } else {
                    if (systemAccum.length() > 0) systemAccum.append('\n');
                    systemAccum.append(lines[i]);
                }
                continue;
            }

            String line = lines[i].strip();
            if (line.isEmpty() || line.startsWith("#")) continue;

            int spaceIdx = line.indexOf(' ');
            if (spaceIdx < 0) continue; // bare word with no argument, ignore

            String instruction = line.substring(0, spaceIdx).toUpperCase(Locale.ROOT);
            String rest = line.substring(spaceIdx + 1).strip();

            switch (instruction) {
                case "FROM":
                    config.setPath(rest);
                    break;

                case "PARAMETER":
                    applyParameter(config, rest);
                    break;

                case "SYSTEM":
                    if (rest.equals("\"\"\"")) {
                        // Multi-line block: collect until closing """
                        systemAccum = new StringBuilder();
                    } else if (rest.startsWith("\"\"\"") && rest.endsWith("\"\"\"") && rest.length() > 6) {
                        // Inline: SYSTEM """text"""
                        config.setSystemPrompt(rest.substring(3, rest.length() - 3).strip());
                    } else {
                        config.setSystemPrompt(rest);
                    }
                    break;

                // TEMPLATE, MESSAGE, ADAPTER, LICENSE: recognised by Ollama but not
                // implemented here; silently ignored so files remain portable.
                default:
                    break;
            }
        }
    }

    private static void applyParameter(ModelConfig config, String paramLine) {
        int spaceIdx = paramLine.indexOf(' ');
        if (spaceIdx < 0) return;
        String key = paramLine.substring(0, spaceIdx).toLowerCase(Locale.ROOT);
        String val = paramLine.substring(spaceIdx + 1).strip();
        try {
            switch (key) {
                case "temperature":  config.setTemperature(Float.parseFloat(val));   break;
                case "num_predict":  config.setNumPredict(Integer.parseInt(val));    break;
                case "num_ctx":      config.setNumCtx(Integer.parseInt(val));        break;
                case "num_thread":
                case "num_threads":  config.setNumThreads(Integer.parseInt(val));    break;
                // top_p, top_k, repeat_penalty, etc. are accepted by Ollama but not
                // yet plumbed through; ignore silently for forward-compat.
            }
        } catch (NumberFormatException e) {
            System.err.println("Warning: Modelfile PARAMETER " + key
                + " has invalid value '" + val + "'; ignored.");
        }
    }

    /**
     * Reconstruct Modelfile text from the fields stored in {@code config}.
     * Only fields that are actually set (non-null) appear in the output.
     */
    public static String toText(ModelConfig config) {
        StringBuilder sb = new StringBuilder();
        if (config.getPath() != null) {
            sb.append("FROM ").append(config.getPath()).append('\n');
        }
        if (config.getTemperature() != null) {
            sb.append("PARAMETER temperature ").append(config.getTemperature()).append('\n');
        }
        if (config.getNumPredict() != null) {
            sb.append("PARAMETER num_predict ").append(config.getNumPredict()).append('\n');
        }
        if (config.getNumCtx() != null) {
            sb.append("PARAMETER num_ctx ").append(config.getNumCtx()).append('\n');
        }
        if (config.getNumThreads() != null) {
            sb.append("PARAMETER num_threads ").append(config.getNumThreads()).append('\n');
        }
        String sys = config.getSystemPrompt();
        if (sys != null && !sys.isEmpty()) {
            if (sys.contains("\n")) {
                sb.append("SYSTEM \"\"\"\n").append(sys);
                if (!sys.endsWith("\n")) sb.append('\n');
                sb.append("\"\"\"\n");
            } else {
                sb.append("SYSTEM ").append(sys).append('\n');
            }
        }
        return sb.toString();
    }
}
