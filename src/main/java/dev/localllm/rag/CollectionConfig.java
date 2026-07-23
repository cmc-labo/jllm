package dev.localllm.rag;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Sidecar metadata recording per-collection customization: whether a RAG collection is
 * hybrid (BM25+vector), which embedding model produced its vectors, and/or a custom
 * chunk size/overlap.
 *
 * <p>Persisted as {@code <indexPath>/config.json}. Its presence is the single source of
 * truth for "has this collection been customized" — collections with no
 * {@code config.json} use every default (plain BM25, {@link DocumentChunker#CHUNK_WORDS}
 * / {@link DocumentChunker#OVERLAP_WORDS}), exactly as before either customization
 * feature existed.
 */
public class CollectionConfig {

    private static final String FILE_NAME = "config.json";
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    /** Registry name of the embedding model used to vectorize this collection, or {@code null}. */
    public String embedModel;

    /** Embedding vector dimensionality, measured from a real {@code embed()} call at index time. */
    public int dimensions;

    /** Custom chunk size in words, or {@code 0} to use {@link DocumentChunker#CHUNK_WORDS}. */
    public int chunkWords;

    /** Custom chunk overlap in words, or {@code 0} to use {@link DocumentChunker#OVERLAP_WORDS}. */
    public int overlapWords;

    public CollectionConfig() {
    }

    public CollectionConfig(String embedModel, int dimensions, int chunkWords, int overlapWords) {
        this.embedModel   = embedModel;
        this.dimensions   = dimensions;
        this.chunkWords   = chunkWords;
        this.overlapWords = overlapWords;
    }

    /** Loads {@code <indexPath>/config.json}, or returns {@code null} if it doesn't exist. */
    public static CollectionConfig load(Path indexPath) {
        Path file = indexPath.resolve(FILE_NAME);
        if (!Files.exists(file)) return null;
        try {
            String json = Files.readString(file);
            return GSON.fromJson(json, CollectionConfig.class);
        } catch (Exception e) {
            return null;
        }
    }

    /** Saves this config to {@code <indexPath>/config.json}, creating the directory if needed. */
    public void save(Path indexPath) throws Exception {
        Files.createDirectories(indexPath);
        Files.writeString(indexPath.resolve(FILE_NAME), GSON.toJson(this));
    }
}
