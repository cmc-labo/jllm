package dev.localllm.plugin;

/**
 * Transforms the fully-assembled ChatML prompt string before it is sent to the model.
 *
 * <h2>Registering an interceptor</h2>
 * Implement this interface in your plugin JAR and create the SPI descriptor:
 * {@code META-INF/services/dev.localllm.plugin.PromptInterceptor}
 * containing the fully-qualified class name. Drop the JAR into
 * {@code ~/.local-llm/plugins/} to activate.
 *
 * <h2>Execution order</h2>
 * Interceptors are sorted by {@link #getPriority()} (ascending) and applied
 * as a pipeline: each interceptor receives the output of the previous one.
 * Use priority 0 for general-purpose interceptors; use negative values to
 * run before built-in processing, positive values to run after.
 *
 * <h2>Scope</h2>
 * Interceptors run in both the interactive REPL ({@code jllm run}) and the
 * embedded HTTP server ({@code jllm serve}).
 *
 * <h2>Thread safety</h2>
 * A single interceptor instance may be called from multiple threads
 * concurrently in server mode. Implementations must be thread-safe.
 */
public interface PromptInterceptor {

    /** Lower numbers run first (0 is a sensible default). */
    int getPriority();

    /**
     * Transform the prompt. Return the prompt unchanged to pass through.
     *
     * @param prompt the assembled ChatML prompt string (may include
     *               {@code <|im_start|>} / {@code <|im_end|>} tags)
     * @return the (potentially modified) prompt; must not be {@code null}
     */
    String intercept(String prompt);
}
