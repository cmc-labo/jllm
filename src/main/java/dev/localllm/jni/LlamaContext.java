package dev.localllm.jni;

import java.lang.ref.Cleaner;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/**
 * An inference context (KV cache + sampler state) bound to a single
 * {@link LlamaModel}. The underlying {@code llama_context} is not
 * reentrant: concurrent {@code llama_decode} calls against the same
 * context corrupt its KV cache / sampler state. For real parallelism
 * across users, create one {@link LlamaContext} per concurrent
 * generation via {@link LlamaModel#createContext}; the model itself is
 * safe to share.
 *
 * <p>If a single context instance is still reached from multiple
 * threads (e.g. pooled or reused), calls into the native context are
 * serialized internally via a lock so they queue safely instead of
 * racing or crashing.
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

    // Guards every native call against this context, so concurrent
    // generation requests on a shared context are serialized rather than
    // racing on llama.cpp's internal KV cache / sampler state, and close()
    // can't free the native handle out from under an in-flight generate().
    private final ReentrantLock lock = new ReentrantLock();

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

    /**
     * Generates up to {@code nPredict} tokens, invoking {@code onToken} as each piece arrives.
     *
     * <p>Thread-safe: if called concurrently on the same context, calls are
     * serialized (one generation runs at a time) rather than corrupting the
     * shared native state. Concurrent requests from different users should
     * still use separate contexts for actual parallelism.
     */
    public void generateStreaming(String prompt, int nPredict, float temperature, Consumer<String> onToken) {
        lock.lock();
        try {
            checkOpen();
            int[] promptTokens = model.tokenize(prompt, true, true);
            LlamaNative.generate(model.handle(), handle, promptTokens, nPredict, temperature, piece -> {
                onToken.accept(piece);
                return true;
            });
        } finally {
            lock.unlock();
        }
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("Context is closed");
        }
    }

    @Override
    public void close() {
        // Block until any in-flight generation finishes before freeing the
        // native handle, to avoid a use-after-free in native code.
        lock.lock();
        try {
            closed = true;
            // Idempotent: safe even if the Cleaner already ran, or close() is
            // called more than once.
            cleanable.clean();
        } finally {
            lock.unlock();
        }
    }
}
