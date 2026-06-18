package dev.localllm.jni;

import java.util.function.Consumer;

/**
 * An inference context (KV cache + sampler state) bound to a single
 * {@link LlamaModel}. Not thread-safe: use one context per concurrent
 * generation.
 */
public final class LlamaContext implements AutoCloseable {

    private final LlamaModel model;
    private final long handle;
    private volatile boolean closed = false;

    LlamaContext(LlamaModel model, long handle) {
        this.model = model;
        this.handle = handle;
    }

    /** Generates up to {@code nPredict} tokens and returns the full text. */
    public String generate(String prompt, int nPredict, float temperature) {
        StringBuilder sb = new StringBuilder();
        generateStreaming(prompt, nPredict, temperature, sb::append);
        return sb.toString();
    }

    /** Generates up to {@code nPredict} tokens, invoking {@code onToken} as each piece arrives. */
    public void generateStreaming(String prompt, int nPredict, float temperature, Consumer<String> onToken) {
        checkOpen();
        int[] promptTokens = model.tokenize(prompt, true, true);
        LlamaNative.generate(model.handle(), handle, promptTokens, nPredict, temperature, piece -> {
            onToken.accept(piece);
            return true;
        });
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("Context is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            LlamaNative.freeContext(handle);
        }
    }
}
