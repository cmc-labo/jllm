package dev.localllm.jni;

import java.lang.ref.Cleaner;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
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

    /**
     * Generates up to {@code nPredict} tokens, exposing them as a pull-based
     * {@link Iterator}/{@link Iterable} instead of a push callback - the
     * natural fit for a thread-per-request blocking server, where each
     * token can be written out as soon as it's pulled rather than buffered
     * until generation finishes.
     *
     * <p>Generation runs on a dedicated background thread that blocks on
     * the native call between tokens, handing each one to the returned
     * {@link TokenStream}. Always {@link TokenStream#close()} it (use
     * try-with-resources) - an abandoned, not-fully-iterated stream would
     * otherwise leak that background thread.
     */
    public TokenStream generateTokens(String prompt, int nPredict, float temperature) {
        checkOpen();
        return new TokenStream(this, prompt, nPredict, temperature);
    }

    /**
     * A single-traversal, pull-based view over the tokens of one
     * {@link #generateTokens} call. Not safe for concurrent use from
     * multiple threads (same as any other {@link Iterator}), but
     * {@link #close()} may be called at any time, from generation start to
     * after exhaustion, to stop generation early and release the
     * background thread.
     */
    public static final class TokenStream implements Iterator<String>, Iterable<String>, AutoCloseable {

        // Sentinel published on the queue once generation has ended (either
        // exhausted nPredict, hit EOG, or was cancelled / errored).
        private static final Object DONE = new Object();

        private final SynchronousQueue<Object> queue = new SynchronousQueue<>();
        private final Thread producer;
        private final AtomicReference<Throwable> error = new AtomicReference<>();

        private boolean lookaheadFetched = false;
        private Object lookahead;
        private boolean closed = false;

        TokenStream(LlamaContext ctx, String prompt, int nPredict, float temperature) {
            producer = new Thread(() -> {
                try {
                    ctx.generateStreaming(prompt, nPredict, temperature, piece -> {
                        try {
                            queue.put(piece);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            // Propagates out of generateStreaming() via the JNI
                            // layer's "callback threw -> stop decoding" check.
                            throw new CancellationException("token stream closed");
                        }
                    });
                } catch (CancellationException expected) {
                    // close() interrupted us to stop generation early.
                } catch (Throwable t) {
                    error.set(t);
                } finally {
                    putUninterruptibly(DONE);
                }
            }, "llama-token-stream");
            producer.setDaemon(true);
            producer.start();
        }

        // Used only for the final DONE handoff: close()'s drain loop always
        // keeps a consumer available until this thread exits, so this never
        // blocks forever in practice - but put() is itself interruptible,
        // so a stray interrupt here must not be allowed to skip the handoff.
        private void putUninterruptibly(Object value) {
            boolean interrupted = false;
            try {
                while (true) {
                    try {
                        queue.put(value);
                        return;
                    } catch (InterruptedException e) {
                        interrupted = true;
                    }
                }
            } finally {
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
            }
        }

        @Override
        public boolean hasNext() {
            fetchLookahead();
            return lookahead != DONE;
        }

        @Override
        public String next() {
            fetchLookahead();
            if (lookahead == DONE) {
                throw new NoSuchElementException();
            }
            String value = (String) lookahead;
            lookahead = null;
            lookaheadFetched = false;
            return value;
        }

        private void fetchLookahead() {
            if (lookaheadFetched) {
                return;
            }
            try {
                lookahead = queue.take();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                close();
                lookahead = DONE;
            }
            lookaheadFetched = true;
            if (lookahead == DONE) {
                Throwable t = error.get();
                if (t instanceof RuntimeException) {
                    throw (RuntimeException) t;
                } else if (t instanceof Error) {
                    throw (Error) t;
                } else if (t != null) {
                    throw new RuntimeException("token generation failed", t);
                }
            }
        }

        @Override
        public Iterator<String> iterator() {
            return this;
        }

        /** Stops generation (if still running) and releases the background thread. Idempotent. */
        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            producer.interrupt();
            // Keep draining so a put() the producer is blocked on (or about
            // to make) always has a consumer, even though nothing else is
            // reading anymore - otherwise it could block forever past the
            // point where interrupting it was supposed to free it.
            while (producer.isAlive()) {
                try {
                    queue.poll(50, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
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
