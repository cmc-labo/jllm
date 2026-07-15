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

    /**
     * Registers the listener that llama.cpp/ggml native log output
     * (model loading, hardware/backend detection, decode warnings, ...)
     * is forwarded to, replacing any previously registered listener.
     * Pass {@code null} to stop forwarding. Must be called before
     * {@link #backendInit()} to also capture logging that happens during
     * backend initialization itself.
     */
    public static native void setLogCallback(LogCallback callback);

    /** ggml_log_level values forwarded to {@link LogCallback#onLog}. */
    public static final int LOG_LEVEL_DEBUG = 1;
    public static final int LOG_LEVEL_INFO  = 2;
    public static final int LOG_LEVEL_WARN  = 3;
    public static final int LOG_LEVEL_ERROR = 4;

    /** Invoked from native code for each complete native log line. */
    public interface LogCallback {
        void onLog(int level, String message);
    }

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

    // ── Continuous-batch primitives ───────────────────────────────────────────

    /**
     * Like {@link #newContext} but sets {@code n_seq_max = nSeqMax}, allowing the
     * context to hold KV cache entries for multiple simultaneous sequences.
     * The total KV cache size is {@code nCtx} tokens shared across all sequences.
     */
    public static native long newBatchContext(long model, int nCtx, int nThreads, int nSeqMax);

    /**
     * Evaluate a batch of tokens.
     *
     * <p>Each element {@code i} describes one token: sequence {@code seqIds[i]},
     * token id {@code tokens[i]}, at KV-cache position {@code positions[i]}.
     * {@code logits} (needed for sampling) are requested for the last token of each
     * consecutive run sharing the same {@code seqId}.
     *
     * @return 0 on success, non-zero on failure
     */
    public static native int batchDecode(long ctx,
                                         int[] seqIds, int[] tokens, int[] positions,
                                         int nTokens);

    /**
     * Create an independent sampler chain (temperature → distribution sampler).
     * Each active sequence must own its own sampler so their internal RNG states
     * do not interfere.
     *
     * @return opaque sampler handle; must be freed with {@link #samplerFree}
     */
    public static native long samplerCreate(float temperature);

    /**
     * Sample the next token for the sequence whose logits are at position
     * {@code batchIdx} in the most recent {@link #batchDecode} output.
     * Pass {@code -1} to sample from the last logits in the batch (useful during
     * single-sequence prefill).
     *
     * @return sampled token id
     */
    public static native int samplerSample(long ctx, long sampler, int batchIdx);

    /** Free a sampler handle created by {@link #samplerCreate}. */
    public static native void samplerFree(long sampler);

    /**
     * Remove KV-cache entries for sequence {@code seqId} in the position range
     * {@code [posFrom, posTo)}.  Pass {@code posTo = -1} to remove all positions
     * from {@code posFrom} to the end.  Used to reclaim KV slots when a sequence
     * finishes.
     */
    public static native void kvSeqRm(long ctx, int seqId, int posFrom, int posTo);

    // ── On-the-fly quantization ───────────────────────────────────────────────

    /**
     * Quantize a GGUF file in-place (i.e. read {@code inputPath}, write the
     * quantized result to {@code outputPath}).  The model is loaded and
     * unloaded internally; no model handle is needed.
     *
     * @param inputPath   path to the source GGUF (F16/BF16/F32 recommended)
     * @param outputPath  path to write the quantized GGUF (must not equal inputPath)
     * @param ftypeInt    {@code llama_ftype} integer value (see {@link QuantizeType#ftypeId})
     * @param nThreads    number of CPU threads to use; ≤0 = hardware concurrency
     * @return 0 on success, non-zero on failure
     */
    public static native int quantize(String inputPath, String outputPath, int ftypeInt, int nThreads);
}
