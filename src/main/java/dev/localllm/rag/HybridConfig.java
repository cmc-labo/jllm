package dev.localllm.rag;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Sidecar metadata recording whether a RAG collection is a hybrid (BM25+vector)
 * collection, and if so which embedding model produced its vectors.
 *
 * <p>Persisted as {@code <indexPath>/hybrid.json}. Its presence is the single source of
 * truth for "is this collection hybrid" — collections with no {@code hybrid.json} are
 * plain BM25, exactly as before this feature existed.
 */
public class HybridConfig {

    private static final String FILE_NAME = "hybrid.json";
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    /** Registry name of the embedding model used to vectorize this collection. */
    public String embedModel;

    /** Embedding vector dimensionality, measured from a real {@code embed()} call at index time. */
    public int dimensions;

    public HybridConfig() {
    }

    public HybridConfig(String embedModel, int dimensions) {
        this.embedModel = embedModel;
        this.dimensions = dimensions;
    }

    /** Loads {@code <indexPath>/hybrid.json}, or returns {@code null} if it doesn't exist. */
    public static HybridConfig load(Path indexPath) {
        Path file = indexPath.resolve(FILE_NAME);
        if (!Files.exists(file)) return null;
        try {
            String json = Files.readString(file);
            return GSON.fromJson(json, HybridConfig.class);
        } catch (Exception e) {
            return null;
        }
    }

    /** Saves this config to {@code <indexPath>/hybrid.json}, creating the directory if needed. */
    public void save(Path indexPath) throws Exception {
        Files.createDirectories(indexPath);
        Files.writeString(indexPath.resolve(FILE_NAME), GSON.toJson(this));
    }
}
