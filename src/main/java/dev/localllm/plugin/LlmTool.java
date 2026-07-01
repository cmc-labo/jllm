package dev.localllm.plugin;

/**
 * A function-calling tool that the model can invoke during a conversation.
 *
 * <h2>Implementing a tool</h2>
 * <ol>
 *   <li>Implement this interface in your plugin JAR.</li>
 *   <li>Register it via Java SPI: create the file<br>
 *       {@code META-INF/services/dev.localllm.plugin.LlmTool}<br>
 *       inside your JAR containing the fully-qualified class name.</li>
 *   <li>Drop the compiled JAR into {@code ~/.local-llm/plugins/}.</li>
 * </ol>
 *
 * <h2>How tool calling works</h2>
 * When at least one tool is loaded, jllm automatically appends a tool-use
 * section to the model's system prompt. The model is instructed to reply with:
 * <pre>
 *   &lt;tool_call&gt;{"name":"TOOL_NAME","args":{...}}&lt;/tool_call&gt;
 * </pre>
 * jllm detects that pattern, calls {@link #execute}, and injects the result
 * back into the conversation before continuing generation.
 *
 * <h2>Thread safety</h2>
 * Each loaded tool instance is shared across all concurrent requests in the
 * HTTP server mode. Implementations must be thread-safe.
 */
public interface LlmTool {

    /**
     * Unique snake_case identifier the model uses to invoke this tool.
     * Must be stable and unique across all loaded plugins (e.g. {@code "weather"}).
     */
    String getName();

    /**
     * One-sentence description inserted into the system prompt so the model
     * understands when to call this tool.
     */
    String getDescription();

    /**
     * JSON Schema string describing the {@code args} object the model should produce.
     * <p>Example:
     * <pre>{"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]}</pre>
     */
    String getParametersSchema();

    /**
     * Execute the tool with the given JSON argument object and return the result
     * as a plain string. The result is injected back into the conversation.
     *
     * @param argsJson a JSON object string matching {@link #getParametersSchema()}
     * @return the tool result (shown verbatim to the model)
     * @throws Exception on error; the exception message is returned to the model as an error result
     */
    String execute(String argsJson) throws Exception;
}
