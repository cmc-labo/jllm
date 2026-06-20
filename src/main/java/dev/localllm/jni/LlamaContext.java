package dev.localllm.jni;

import java.lang.ref.Cleaner;
import java.util.function.Consumer;

/**
 * An inference context (KV cache + sampler state) bound to a single
 * {@link LlamaModel}. Not thread-safe: use one context per concurrent
 * generation.
 *
 * <p>Like {@link LlamaModel}, the native context lives in off-heap memory.
 * Always {@link #close()} (or use try-with-resources); a {@link Cleaner}
 * acts only as a best-effort safety net for leaked instances.
 */
public final class LlamaContext implements AutoCloseable {

    /**
     * Holds only the native handle - must not (transitively) reference the
     * enclosing LlamaContext, or the instance would never become unreachable
     * and the Cleaner would never run.
     */
    private static final class State implements Runnable {
        private final long handle;

        State(long handle) {
            this.handle = handle;
        }

        @Override
        public void run() {
            LlamaNative.freeContext(handle);
        }
    }

    private final LlamaModel model;
    private final long handle;
    private final Cleaner.Cleanable cleanable;
    private volatile boolean closed = false;

    LlamaContext(LlamaModel model, long handle) {
        this.model = model;
        this.handle = handle;
        this.cleanable = NativeCleaner.INSTANCE.register(this, new State(handle));
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
        closed = true;
        // Idempotent: safe even if the Cleaner already ran, or close() is
        // called more than once.
        cleanable.clean();
    }
}
