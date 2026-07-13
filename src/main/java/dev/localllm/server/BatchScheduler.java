package dev.localllm.server;

import dev.localllm.jni.BatchContext;
import dev.localllm.jni.LlamaModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Simplified continuous-batch scheduler.
 *
 * <h2>Design</h2>
 * A single daemon thread drives the decode loop:
 * <ol>
 *   <li><b>Prefill</b> — for each pending request that fits in a free slot,
 *       run the full prompt through {@link BatchContext#batchDecode}, sample the
 *       first token, and add the slot to the active list.</li>
 *   <li><b>Decode</b> — submit one token per active sequence in a single
 *       {@link BatchContext#batchDecode} call, then sample one token per sequence.
 *       This is the "continuous batching" step that amortises GPU kernel launch
 *       overhead across all active sequences.</li>
 * </ol>
 *
 * <h2>KV-cache</h2>
 * Each active sequence gets its own {@code seq_id} (0 .. maxSeqs-1).  llama.cpp
 * stores KV entries as (seq_id, position) pairs, so two sequences with the same
 * {@code position} value do not collide: attention is filtered by {@code seq_id}.
 * The shared KV cache has {@code nCtxPerSeq * maxSeqs} total slots.
 *
 * <h2>Concurrency</h2>
 * All llama.cpp calls happen on the scheduler thread.  Callers submit requests
 * via {@link #submit} and receive tokens through a {@link Sequence} handle, which
 * exposes a blocking {@link Sequence#nextPiece()} method safe to call from any thread.
 */
public final class BatchScheduler {

    private static final Logger LOG = LoggerFactory.getLogger(BatchScheduler.class);

    // ── Public API types ──────────────────────────────────────────────────────

    /**
     * Token wrapper.  {@code text == null} is the end-of-generation sentinel
     * (cannot use null directly in {@link LinkedBlockingQueue}).
     */
    public static final class Piece {
        static final Piece END = new Piece(null);
        public final String text;
        Piece(String text) { this.text = text; }
    }

    /**
     * Handle returned to the caller of {@link #submit}.
     * Call {@link #nextPiece()} in a loop until it returns {@code null}.
     */
    public static final class Sequence {
        private final LinkedBlockingQueue<Piece> queue;
        Sequence(LinkedBlockingQueue<Piece> queue) { this.queue = queue; }

        /** Blocks until the next token arrives.  Returns {@code null} when done. */
        public String nextPiece() throws InterruptedException {
            return queue.take().text;
        }
    }

    // ── Internal types ────────────────────────────────────────────────────────

    private static final class PendingRequest {
        final int[]   promptTokens;
        final int     nPredict;
        final float   temperature;
        final LinkedBlockingQueue<Piece> output = new LinkedBlockingQueue<>();

        PendingRequest(int[] promptTokens, int nPredict, float temperature) {
            this.promptTokens = promptTokens;
            this.nPredict     = nPredict;
            this.temperature  = temperature;
        }
    }

    private static final class SeqSlot {
        final PendingRequest req;
        final int seqId;
        final int maxPosition;  // max KV tokens this sequence may occupy
        long  sampler;
        int   position;         // next KV position to write (= promptLen after prefill)
        int   lastToken;        // token to feed on the next decode step
        int   nDecoded;         // tokens generated so far (including the prefill sample)

        SeqSlot(PendingRequest req, int seqId, int maxPosition,
                long sampler, int position, int lastToken) {
            this.req         = req;
            this.seqId       = seqId;
            this.maxPosition = maxPosition;
            this.sampler     = sampler;
            this.position    = position;
            this.lastToken   = lastToken;
            this.nDecoded    = 1; // prefill already produced one token
        }
    }

    // ── Fields ────────────────────────────────────────────────────────────────

    private final LlamaModel model;
    private final BatchContext batchCtx;
    private final int maxSeqs;
    private final int seqSlotSize; // nCtxPerSeq

    private final int[] freeSlotIds; // stack of available seq IDs
    private int freeTop;

    private final List<SeqSlot> activeSlots = new ArrayList<>();
    private final LinkedBlockingQueue<PendingRequest> pendingQueue;

    private volatile boolean running = true;
    private final Thread schedulerThread;

    // ── Constructor ───────────────────────────────────────────────────────────

    public BatchScheduler(LlamaModel model, int nCtxPerSeq, int nThreads, int maxSeqs) {
        this.model        = model;
        this.maxSeqs      = maxSeqs;
        this.seqSlotSize  = nCtxPerSeq;
        this.batchCtx     = model.createBatchContext(nCtxPerSeq, nThreads, maxSeqs);
        this.pendingQueue = new LinkedBlockingQueue<>(maxSeqs * 8);

        this.freeSlotIds = new int[maxSeqs];
        for (int i = 0; i < maxSeqs; i++) freeSlotIds[i] = i;
        this.freeTop = maxSeqs;

        this.schedulerThread = new Thread(this::schedulerLoop, "batch-scheduler");
        this.schedulerThread.setDaemon(true);
        this.schedulerThread.start();
    }

    // ── Public interface ──────────────────────────────────────────────────────

    /**
     * Enqueue a generation request and return a {@link Sequence} handle.
     * Blocks if the internal pending queue is full (back-pressure).
     */
    public Sequence submit(int[] promptTokens, int nPredict, float temperature)
            throws InterruptedException {
        PendingRequest req = new PendingRequest(promptTokens, nPredict, temperature);
        pendingQueue.put(req);
        return new Sequence(req.output);
    }

    /** Gracefully stop the scheduler thread. */
    public void shutdown() {
        running = false;
        schedulerThread.interrupt();
    }

    public int activeSequences()  { return activeSlots.size(); }
    public int pendingRequests()  { return pendingQueue.size(); }

    // ── Scheduler loop (runs on schedulerThread) ──────────────────────────────

    private void schedulerLoop() {
        while (running) {
            try {
                if (activeSlots.isEmpty()) {
                    // Idle: block until at least one request arrives.
                    PendingRequest req = pendingQueue.poll(20, TimeUnit.MILLISECONDS);
                    if (req == null) continue;
                    prefillAndActivate(req);
                    if (activeSlots.isEmpty()) continue;
                }

                // Promote any waiting requests into free slots (non-blocking).
                while (freeTop > 0) {
                    PendingRequest req = pendingQueue.peek();
                    if (req == null) break;
                    pendingQueue.poll();
                    prefillAndActivate(req);
                }

                if (!activeSlots.isEmpty()) {
                    decodeStep();
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                LOG.error("BatchScheduler: unexpected error in scheduler loop", e);
                // Abort all active sequences, then keep running for future requests.
                for (SeqSlot slot : activeSlots) {
                    slot.req.output.offer(Piece.END);
                    cleanupSlot(slot);
                }
                activeSlots.clear();
            }
        }

        // Drain pending queue on shutdown.
        PendingRequest req;
        while ((req = pendingQueue.poll()) != null) req.output.offer(Piece.END);
        for (SeqSlot slot : activeSlots) {
            slot.req.output.offer(Piece.END);
            cleanupSlot(slot);
        }
        activeSlots.clear();
        batchCtx.close();
    }

    // ── Prefill ───────────────────────────────────────────────────────────────

    private void prefillAndActivate(PendingRequest req) {
        if (freeTop == 0) {
            // No free slot — re-enqueue (shouldn't normally happen).
            try { pendingQueue.put(req); } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                req.output.offer(Piece.END);
            }
            return;
        }

        int promptLen = req.promptTokens.length;
        if (promptLen == 0 || promptLen > seqSlotSize) {
            // Prompt empty or exceeds per-sequence context budget.
            req.output.offer(Piece.END);
            return;
        }

        int seqId = freeSlotIds[--freeTop];

        // Build prefill batch (all tokens in the prompt for this one sequence).
        int[] seqIds    = new int[promptLen];
        int[] tokens    = new int[promptLen];
        int[] positions = new int[promptLen];
        for (int i = 0; i < promptLen; i++) {
            seqIds[i]    = seqId;
            tokens[i]    = req.promptTokens[i];
            positions[i] = i;
        }

        long sampler = batchCtx.samplerCreate(req.temperature);
        if (sampler == 0) {
            freeSlotIds[freeTop++] = seqId;
            req.output.offer(Piece.END);
            return;
        }

        int ret = batchCtx.batchDecode(seqIds, tokens, positions, promptLen);
        if (ret != 0) {
            LOG.warn("BatchScheduler: prefill decode failed (ret={})", ret);
            batchCtx.samplerFree(sampler);
            batchCtx.kvSeqRm(seqId, 0, -1);
            freeSlotIds[freeTop++] = seqId;
            req.output.offer(Piece.END);
            return;
        }

        // Sample the first generated token from the last prefill logits.
        int firstToken = batchCtx.samplerSample(sampler, -1);
        if (model.isEog(firstToken)) {
            batchCtx.samplerFree(sampler);
            batchCtx.kvSeqRm(seqId, 0, -1);
            freeSlotIds[freeTop++] = seqId;
            req.output.offer(Piece.END);
            return;
        }

        req.output.offer(new Piece(model.tokenToPiece(firstToken)));
        activeSlots.add(new SeqSlot(req, seqId, seqSlotSize, sampler, promptLen, firstToken));
    }

    // ── Decode step ───────────────────────────────────────────────────────────

    private void decodeStep() {
        int n = activeSlots.size();
        int[] seqIds    = new int[n];
        int[] tokens    = new int[n];
        int[] positions = new int[n];

        for (int i = 0; i < n; i++) {
            SeqSlot slot = activeSlots.get(i);
            seqIds[i]    = slot.seqId;
            tokens[i]    = slot.lastToken;
            positions[i] = slot.position;
        }

        int ret = batchCtx.batchDecode(seqIds, tokens, positions, n);
        if (ret != 0) {
            LOG.warn("BatchScheduler: decode step failed (ret={}), aborting {} sequence(s)", ret, n);
            for (SeqSlot slot : activeSlots) {
                slot.req.output.offer(Piece.END);
                cleanupSlot(slot);
            }
            activeSlots.clear();
            return;
        }

        List<SeqSlot> finished = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            SeqSlot slot = activeSlots.get(i);
            int newToken = batchCtx.samplerSample(slot.sampler, i);
            slot.position++;
            slot.nDecoded++;

            boolean done = model.isEog(newToken)
                        || slot.nDecoded >= slot.req.nPredict
                        || slot.position >= slot.maxPosition;

            if (done) {
                slot.req.output.offer(Piece.END);
                cleanupSlot(slot);
                finished.add(slot);
            } else {
                slot.lastToken = newToken;
                slot.req.output.offer(new Piece(model.tokenToPiece(newToken)));
            }
        }
        activeSlots.removeAll(finished);
    }

    // ── Slot cleanup ──────────────────────────────────────────────────────────

    private void cleanupSlot(SeqSlot slot) {
        batchCtx.samplerFree(slot.sampler);
        slot.sampler = 0;
        batchCtx.kvSeqRm(slot.seqId, 0, -1);
        freeSlotIds[freeTop++] = slot.seqId;
    }
}
