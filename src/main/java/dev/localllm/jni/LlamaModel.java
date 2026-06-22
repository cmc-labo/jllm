package dev.localllm.jni;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A loaded GGUF model. Thread-safe to share across contexts; not safe to use
 * after {@link #close()}.
 *
 * <p>The native model occupies off-heap memory the JVM's GC does not know
 * about. Always {@link #close()} (or use try-with-resources) to free it
 * deterministically. As a safety net for leaked instances, a {@link Cleaner}
 * also frees the native handle when this object becomes unreachable - but
 * that only runs at GC's discretion and must not be relied upon.
 */
public final class LlamaModel implements AutoCloseable {

    private static final AtomicBoolean BACKEND_INITIALIZED = new AtomicBoolean(false);

    /**
     * Holds only the native handle - must not (transitively) reference the
     * enclosing LlamaModel, or the instance would never become unreachable
     * and the Cleaner would never run.
     */
    private static final class State implements Runnable {
        private final long handle;

        State(long handle) {
            this.handle = handle;
        }

        @Override
        public void run() {
            LlamaNative.freeModel(handle);
        }
    }

    private final long handle;
    private final Cleaner.Cleanable cleanable;
    private volatile boolean closed = false;

    public LlamaModel(String path) {
        this(path, 0);
    }

    /** @param nGpuLayers number of layers to offload to GPU; 0 = CPU only, negative = all layers. */
    public LlamaModel(String path, int nGpuLayers) {
        if (BACKEND_INITIALIZED.compareAndSet(false, true)) {
            // Install the log bridge before backendInit() so logging from
            // backend/hardware detection itself is captured too.
            NativeLogBridge.installOnce();
            LlamaNative.backendInit();
        }
        this.handle = LlamaNative.loadModel(path, nGpuLayers);
        if (this.handle == 0) {
            throw new IllegalStateException("Failed to load model: " + path);
        }
        this.cleanable = NativeCleaner.INSTANCE.register(this, new State(this.handle));
    }

    public LlamaContext createContext(int nCtx, int nThreads) {
        checkOpen();
        long ctxHandle = LlamaNative.newContext(handle, nCtx, nThreads);
        if (ctxHandle == 0) {
            throw new IllegalStateException("Failed to create context (nCtx=" + nCtx + ")");
        }
        return new LlamaContext(this, ctxHandle);
    }

    public int[] tokenize(String text, boolean addSpecial, boolean parseSpecial) {
        checkOpen();
        return LlamaNative.tokenize(handle, text, addSpecial, parseSpecial);
    }

    public String tokenToPiece(int token) {
        checkOpen();
        return LlamaNative.tokenToPiece(handle, token);
    }

    public boolean isEog(int token) {
        checkOpen();
        return LlamaNative.isEog(handle, token);
    }

    long handle() {
        checkOpen();
        return handle;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("Model is closed");
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
