package dev.localllm.jni;

/**
 * A multi-sequence llama.cpp context backed by a shared KV cache.
 * All operations delegate to {@link LlamaNative}; callers are responsible
 * for single-threaded access (the {@link dev.localllm.server.BatchScheduler}
 * owns all calls on its scheduler thread).
 */
public final class BatchContext implements AutoCloseable {

    private volatile long handle;

    BatchContext(long handle) {
        this.handle = handle;
    }

    /**
     * Decode a mixed batch of tokens.
     * Each element {@code i} belongs to sequence {@code seqIds[i]} and occupies
     * KV-cache position {@code positions[i]}.
     *
     * @return 0 on success, non-zero on failure
     */
    public int batchDecode(int[] seqIds, int[] tokens, int[] positions, int nTokens) {
        return LlamaNative.batchDecode(handle, seqIds, tokens, positions, nTokens);
    }

    /**
     * Create an independent sampler chain for one sequence.
     * Must be freed with {@link #samplerFree} when the sequence finishes.
     *
     * @return opaque sampler handle, or 0 on failure
     */
    public long samplerCreate(float temperature) {
        return LlamaNative.samplerCreate(temperature);
    }

    /**
     * Sample the next token for the sequence whose logits sit at {@code batchIdx}
     * in the most recent {@link #batchDecode} output.
     * Pass {@code -1} to sample from the last logits in the batch (prefill path).
     */
    public int samplerSample(long samplerHandle, int batchIdx) {
        return LlamaNative.samplerSample(handle, samplerHandle, batchIdx);
    }

    /** Free a sampler created by {@link #samplerCreate}. Null-safe (0 is ignored). */
    public void samplerFree(long samplerHandle) {
        if (samplerHandle != 0) LlamaNative.samplerFree(samplerHandle);
    }

    /**
     * Remove KV-cache entries for {@code seqId} in {@code [posFrom, posTo)}.
     * Pass {@code posTo = -1} to remove from {@code posFrom} to the end.
     */
    public void kvSeqRm(int seqId, int posFrom, int posTo) {
        LlamaNative.kvSeqRm(handle, seqId, posFrom, posTo);
    }

    @Override
    public void close() {
        long h = handle;
        handle = 0;
        if (h != 0) LlamaNative.freeContext(h);
    }
}
