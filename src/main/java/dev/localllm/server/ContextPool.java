package dev.localllm.server;

import dev.localllm.jni.LlamaContext;
import dev.localllm.jni.LlamaModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Per-configuration pool of reusable {@link LlamaContext} instances.
 *
 * <h2>Why pooling helps</h2>
 * Every {@link LlamaContext} owns a KV cache buffer allocated in native (off-heap)
 * memory. For a 3B model at nCtx=4096 this is roughly 400–600 MB; for a 70B model
 * it can exceed several GB. Allocating and freeing that buffer on every request
 * (the previous behaviour) adds non-trivial latency and causes memory fragmentation
 * in the native allocator.
 *
 * <h2>Safety of reuse</h2>
 * {@link LlamaContext#generateTokens} always decodes the <em>full</em> prompt token
 * sequence starting at KV-cache position 0, overwriting any previous entries. Stale
 * entries from a previous request beyond the new prompt's length are never read, so
 * reusing a context without explicit {@code llama_kv_cache_clear} is safe.
 *
 * <h2>Pool key</h2>
 * {@code "modelName|nCtx|nThreads"} — a context with nCtx=4096 cannot satisfy a
 * request that needs nCtx=8192, so the key must encode both dimensions.
 *
 * <h2>Capacity</h2>
 * At most {@code maxPerKey} idle contexts are kept per key; extras are closed
 * immediately on release. Setting {@code maxPerKey = maxConcurrent} means the pool
 * can absorb all contexts that could be in-flight simultaneously, so no context is
 * ever evicted under normal load.
 */
public class ContextPool implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(ContextPool.class);

    // idle contexts, keyed by "modelName|nCtx|nThreads"
    private final ConcurrentHashMap<String, ConcurrentLinkedDeque<LlamaContext>> pool
            = new ConcurrentHashMap<>();

    private final int       maxPerKey;
    private final AtomicLong hits      = new AtomicLong();
    private final AtomicLong misses    = new AtomicLong();
    private final AtomicLong evictions = new AtomicLong();

    public ContextPool(int maxPerKey) {
        this.maxPerKey = maxPerKey;
    }

    // ── Core operations ───────────────────────────────────────────────────────

    /**
     * Acquire a context for the given model configuration.
     * Returns a pooled context (cache hit) if one is available, otherwise allocates
     * a fresh context (cache miss). The caller must later call {@link #release}.
     */
    public LlamaContext acquire(LlamaModel model, String modelName, int nCtx, int nThreads) {
        String key = key(modelName, nCtx, nThreads);
        ConcurrentLinkedDeque<LlamaContext> queue = pool.get(key);
        if (queue != null) {
            LlamaContext ctx = queue.poll();
            if (ctx != null) {
                hits.incrementAndGet();
                LOG.debug("ctx-pool HIT  {} (idle={} remaining)", key, queue.size());
                return ctx;
            }
        }
        misses.incrementAndGet();
        LOG.debug("ctx-pool MISS {} — allocating", key);
        return model.createContext(nCtx, nThreads);
    }

    /**
     * Return a context to the pool after use. If the pool for this key already
     * holds {@code maxPerKey} idle contexts, the context is closed instead.
     *
     * <p>Always call this from a {@code finally} block after calling
     * {@link #acquire}, so the context is never permanently leaked.
     */
    public void release(String modelName, int nCtx, int nThreads, LlamaContext ctx) {
        if (ctx == null) return;
        String key = key(modelName, nCtx, nThreads);
        ConcurrentLinkedDeque<LlamaContext> queue =
                pool.computeIfAbsent(key, k -> new ConcurrentLinkedDeque<>());
        if (queue.size() < maxPerKey) {
            queue.offer(ctx);
            LOG.debug("ctx-pool RETURN {} (idle={})", key, queue.size());
        } else {
            evictions.incrementAndGet();
            LOG.debug("ctx-pool FULL  {} — closing evicted context", key);
            ctx.close();
        }
    }

    /**
     * Close and discard all pooled contexts for the given model.
     * Call this when a model is unregistered or its configuration changes.
     */
    public void invalidate(String modelName) {
        pool.entrySet().removeIf(entry -> {
            if (!entry.getKey().startsWith(modelName + "|")) return false;
            LlamaContext ctx;
            while ((ctx = entry.getValue().poll()) != null) ctx.close();
            LOG.debug("ctx-pool invalidated entries for model '{}'", modelName);
            return true;
        });
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    /** Snapshot of pool metrics for monitoring and logging. */
    public PoolStats stats() {
        List<KeyStats> keys = new ArrayList<>();
        long totalIdle = 0;
        for (Map.Entry<String, ConcurrentLinkedDeque<LlamaContext>> e : pool.entrySet()) {
            int idle = e.getValue().size();
            totalIdle += idle;
            if (idle > 0) keys.add(new KeyStats(e.getKey(), idle));
        }
        return new PoolStats(hits.get(), misses.get(), evictions.get(), totalIdle, pool.size(), keys);
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────

    /** Close all pooled contexts. Call on server shutdown. */
    @Override
    public void close() {
        pool.forEach((key, queue) -> {
            LlamaContext ctx;
            while ((ctx = queue.poll()) != null) ctx.close();
        });
        pool.clear();
        LOG.debug("ctx-pool shut down");
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private static String key(String modelName, int nCtx, int nThreads) {
        return modelName + '|' + nCtx + '|' + nThreads;
    }

    // ── Stats types ───────────────────────────────────────────────────────────

    public static final class KeyStats {
        public final String key;
        public final int    idleContexts;
        KeyStats(String key, int idleContexts) { this.key = key; this.idleContexts = idleContexts; }
    }

    public static final class PoolStats {
        public final long          hits;
        public final long          misses;
        public final long          evictions;
        public final long          totalIdle;
        public final int           distinctKeys;
        public final List<KeyStats> byKey;

        PoolStats(long hits, long misses, long evictions,
                  long totalIdle, int distinctKeys, List<KeyStats> byKey) {
            this.hits         = hits;
            this.misses       = misses;
            this.evictions    = evictions;
            this.totalIdle    = totalIdle;
            this.distinctKeys = distinctKeys;
            this.byKey        = byKey;
        }

        public double hitRate() {
            long total = hits + misses;
            return total == 0 ? 0.0 : (double) hits / total * 100.0;
        }

        @Override
        public String toString() {
            return String.format(
                "hits=%d  misses=%d  hit_rate=%.1f%%  idle=%d  keys=%d  evictions=%d",
                hits, misses, hitRate(), totalIdle, distinctKeys, evictions);
        }
    }
}
