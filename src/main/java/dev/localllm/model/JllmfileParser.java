package dev.localllm.model;

import java.util.Locale;

/**
 * Minimal YAML-based Jllmfile parser.
 *
 * <p>Supported top-level fields:
 * <pre>
 *   from: /path/to/model.gguf
 *   binary: /usr/local/bin/llama-cli
 *   system: "Single-line system prompt."
 *   system: |
 *     Multi-line
 *     system prompt.
 *   parameters:
 *     temperature: 0.7
 *     num_predict: 1024
 *     num_ctx: 4096
 *     num_threads: 4
 * </pre>
 *
 * <p>Comments ({@code #}) and blank lines are ignored. Unknown keys are
 * silently skipped so future fields can be added without breaking old parsers.
 */
public final class JllmfileParser {

    private JllmfileParser() {}

    /**
     * Parse {@code content} and apply recognised fields to {@code config}.
     * Existing fields not covered by the Jllmfile are left unchanged.
     */
    public static void apply(String content, ModelConfig config) {
        String[] lines = content.split("\\r?\\n", -1);

        boolean inParameters = false;
        boolean inSystemBlock = false;   // inside a `system: |` block scalar
        int systemIndent = -1;           // leading-space count of the block body
        StringBuilder systemAccum = null;

        for (String raw : lines) {
            // ── block-scalar body ────────────────────────────────────────────
            if (inSystemBlock) {
                int indent = leadingSpaces(raw);
                String trimmed = raw.strip();

                // blank lines inside a block scalar are kept
                if (trimmed.isEmpty()) {
                    systemAccum.append('\n');
                    continue;
                }

                // a line with less indentation than the first content line ends the block
                if (systemIndent < 0) {
                    systemIndent = indent; // first non-blank content line sets the indent
                    systemAccum.append(raw.substring(systemIndent)).append('\n');
                    continue;
                } else if (indent < systemIndent) {
                    config.setSystemPrompt(systemAccum.toString().stripTrailing());
                    systemAccum = null;
                    inSystemBlock = false;
                    inParameters = false;
                    // fall through and re-parse this line as a root-level key
                } else {
                    systemAccum.append(raw.substring(systemIndent)).append('\n');
                    continue;
                }
            }

            String line = raw.strip();
            if (line.isEmpty() || line.startsWith("#")) continue;

            // ── parameters sub-block ─────────────────────────────────────────
            if (inParameters) {
                if (leadingSpaces(raw) == 0 && !raw.startsWith(" ") && !raw.startsWith("\t")) {
                    // dedented → back to root level
                    inParameters = false;
                } else {
                    applyParameter(config, line);
                    continue;
                }
            }

            // ── root-level key: value ─────────────────────────────────────────
            int colon = line.indexOf(':');
            if (colon < 0) continue;

            String key = line.substring(0, colon).strip().toLowerCase(Locale.ROOT);
            String value = line.substring(colon + 1).strip();

            switch (key) {
                case "from":
                    if (!value.isEmpty()) config.setPath(value);
                    break;

                case "binary":
                    if (!value.isEmpty()) config.setBinary(value);
                    break;

                case "system":
                    if (value.equals("|")) {
                        // block scalar — collect following indented lines
                        inSystemBlock = true;
                        systemIndent  = -1;
                        systemAccum   = new StringBuilder();
                    } else {
                        config.setSystemPrompt(unquote(value));
                    }
                    break;

                case "parameters":
                    inParameters = true;
                    break;

                default:
                    break;
            }
        }

        // handle a system block that reaches EOF without dedenting
        if (inSystemBlock && systemAccum != null) {
            config.setSystemPrompt(systemAccum.toString().stripTrailing());
        }
    }

    private static void applyParameter(ModelConfig config, String line) {
        int colon = line.indexOf(':');
        if (colon < 0) return;
        String key = line.substring(0, colon).strip().toLowerCase(Locale.ROOT);
        String val = line.substring(colon + 1).strip();
        if (val.isEmpty()) return;
        try {
            switch (key) {
                case "temperature":  config.setTemperature(Float.parseFloat(val));   break;
                case "num_predict":  config.setNumPredict(Integer.parseInt(val));    break;
                case "num_ctx":      config.setNumCtx(Integer.parseInt(val));        break;
                case "num_thread":
                case "num_threads":  config.setNumThreads(Integer.parseInt(val));    break;
            }
        } catch (NumberFormatException e) {
            System.err.println("Warning: Jllmfile parameter '" + key
                + "' has invalid value '" + val + "'; ignored.");
        }
    }

    /** Strip surrounding double- or single-quotes from a YAML scalar value. */
    private static String unquote(String s) {
        if (s.length() >= 2
                && ((s.startsWith("\"") && s.endsWith("\""))
                    || (s.startsWith("'")  && s.endsWith("'")))) {
            return s.substring(1, s.length() - 1);
        }
        return s;
    }

    private static int leadingSpaces(String line) {
        int count = 0;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == ' ')       { count++;     }
            else if (c == '\t') { count += 2;  } // treat tab as 2 spaces
            else                { break;        }
        }
        return count;
    }

    /**
     * Reconstruct Jllmfile YAML text from the fields stored in {@code config}.
     * Only non-null fields are emitted.
     */
    public static String toText(ModelConfig config) {
        StringBuilder sb = new StringBuilder();
        if (config.getPath() != null) {
            sb.append("from: ").append(config.getPath()).append('\n');
        }
        if (config.getBinary() != null) {
            sb.append("binary: ").append(config.getBinary()).append('\n');
        }
        String sys = config.getSystemPrompt();
        if (sys != null && !sys.isEmpty()) {
            if (sys.contains("\n")) {
                sb.append("system: |\n");
                for (String l : sys.split("\\n", -1)) {
                    sb.append("  ").append(l).append('\n');
                }
            } else {
                sb.append("system: \"").append(sys.replace("\"", "\\\"")).append("\"\n");
            }
        }
        boolean hasParam = config.getTemperature() != null || config.getNumPredict() != null
                        || config.getNumCtx()      != null || config.getNumThreads() != null;
        if (hasParam) {
            sb.append("parameters:\n");
            if (config.getTemperature()  != null) sb.append("  temperature: ").append(config.getTemperature()).append('\n');
            if (config.getNumPredict()   != null) sb.append("  num_predict: ").append(config.getNumPredict()).append('\n');
            if (config.getNumCtx()       != null) sb.append("  num_ctx: ").append(config.getNumCtx()).append('\n');
            if (config.getNumThreads()   != null) sb.append("  num_threads: ").append(config.getNumThreads()).append('\n');
        }
        return sb.toString();
    }
}
