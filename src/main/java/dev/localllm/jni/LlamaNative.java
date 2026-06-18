package dev.localllm.jni;

/**
 * Raw JNI bindings to llama.cpp's C API. All handles are native pointers
 * boxed as {@code long}. Prefer {@link LlamaModel} / {@link LlamaContext}
 * over calling this class directly.
 */
public final class LlamaNative {

    static {
        NativeLibraryLoader.load();
    }

    private LlamaNative() {}

    public static native void backendInit();
    public static native void backendFree();

    /** Returns a model handle, or 0 on failure. */
    public static native long loadModel(String path, int nGpuLayers);
    public static native void freeModel(long model);

    /** Returns a context handle, or 0 on failure. */
    public static native long newContext(long model, int nCtx, int nThreads);
    public static native void freeContext(long ctx);

    public static native int[] tokenize(long model, String text, boolean addSpecial, boolean parseSpecial);
    public static native String tokenToPiece(long model, int token);
    public static native boolean isEog(long model, int token);

    /**
     * Feeds {@code promptTokens} through the model and greedily/temperature-samples
     * up to {@code nPredict} further tokens, invoking {@code callback} once per
     * generated token. Returns the number of tokens generated.
     */
    public static native int generate(
        long model,
        long ctx,
        int[] promptTokens,
        int nPredict,
        float temperature,
        TokenCallback callback
    );

    /** Invoked from native code for each generated token; return false to stop early. */
    public interface TokenCallback {
        boolean onToken(String piece);
    }
}
