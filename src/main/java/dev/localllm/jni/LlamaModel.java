package dev.localllm.jni;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A loaded GGUF model. Thread-safe to share across contexts; not safe to use
 * after {@link #close()}.
 */
public final class LlamaModel implements AutoCloseable {

    private static final AtomicBoolean BACKEND_INITIALIZED = new AtomicBoolean(false);

    private final long handle;
    private volatile boolean closed = false;

    public LlamaModel(String path) {
        this(path, 0);
    }

    /** @param nGpuLayers number of layers to offload to GPU; 0 = CPU only, negative = all layers. */
    public LlamaModel(String path, int nGpuLayers) {
        if (BACKEND_INITIALIZED.compareAndSet(false, true)) {
            LlamaNative.backendInit();
        }
        this.handle = LlamaNative.loadModel(path, nGpuLayers);
        if (this.handle == 0) {
            throw new IllegalStateException("Failed to load model: " + path);
        }
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
        if (!closed) {
            closed = true;
            LlamaNative.freeModel(handle);
        }
    }
}
