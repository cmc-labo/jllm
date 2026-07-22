package dev.localllm.rag;

/**
 * A single chunk retrieved from a Lucene RAG index.
 *
 * <p>{@link #score} is the ranking score used to order results: the raw BM25 score for
 * BM25-only collections, or the fused Reciprocal Rank Fusion score for hybrid
 * (BM25+vector) collections. {@link #bm25Score} and {@link #vectorScore} are the
 * per-signal scores that contributed to the fusion — {@link Float#NaN} when a signal
 * did not contribute (e.g. {@code vectorScore} on a BM25-only collection, or either
 * score for a result that only appeared in one of the two ranked lists).
 */
public class RagResult {

    /** Absolute path of the source document. */
    public final String source;

    /** 1-based page number for PDFs; -1 for plain-text files (no page concept). */
    public final int page;

    /** Text content of the chunk. */
    public final String content;

    /** Ranking score (BM25 score, or fused RRF score for hybrid collections). */
    public final float score;

    /** Raw BM25 score, or {@link Float#NaN} if this result didn't come from the BM25 side. */
    public final float bm25Score;

    /** Cosine similarity score, or {@link Float#NaN} if this result didn't come from the vector side. */
    public final float vectorScore;

    public RagResult(String source, int page, String content, float score) {
        this(source, page, content, score, Float.NaN, Float.NaN);
    }

    public RagResult(String source, int page, String content, float score,
                      float bm25Score, float vectorScore) {
        this.source      = source;
        this.page        = page;
        this.content     = content;
        this.score       = score;
        this.bm25Score   = bm25Score;
        this.vectorScore = vectorScore;
    }
}
