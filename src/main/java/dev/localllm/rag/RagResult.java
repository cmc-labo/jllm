package dev.localllm.rag;

/** A single chunk retrieved from a Lucene RAG index, with its BM25 relevance score. */
public class RagResult {

    /** Absolute path of the source document. */
    public final String source;

    /** 1-based page number for PDFs; -1 for plain-text files (no page concept). */
    public final int page;

    /** Text content of the chunk. */
    public final String content;

    /** BM25 relevance score (higher = more relevant). */
    public final float score;

    public RagResult(String source, int page, String content, float score) {
        this.source  = source;
        this.page    = page;
        this.content = content;
        this.score   = score;
    }
}
