package dev.localllm.rag;

import dev.localllm.jni.LlamaModel;
import dev.localllm.model.ModelConfig;
import dev.localllm.model.ModelRegistry;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * High-level manager for local RAG (Retrieval-Augmented Generation) collections.
 *
 * <p>Each collection is a named Lucene index stored at
 * {@code <ragDir>/<collectionName>/}.  Documents (PDFs or text files) are split
 * into overlapping word-level chunks by {@link DocumentChunker} and indexed for
 * BM25 full-text search by {@link RagIndex}.
 *
 * <p>A collection can optionally be made <b>hybrid</b> (BM25 + vector search) by
 * indexing it with an embedding model: {@code jllm rag add <collection> <path>
 * --embed-model <name>}. This is opt-in per collection — collections indexed without
 * {@code --embed-model} behave exactly as before (pure BM25, no embedding model loaded,
 * zero extra cost). Whether a collection is hybrid is recorded in {@link HybridConfig}
 * and read back automatically on subsequent {@code add}/{@code search} calls.
 *
 * <p>Usage flow:
 * <ol>
 *   <li>Index documents: {@code jllm rag add <collection> <path> [--embed-model <name>]}</li>
 *   <li>Chat with RAG: {@code jllm run <model> --rag <collection>}</li>
 *   <li>Or via API: include {@code "rag_collection": "<name>"} in the request body.</li>
 * </ol>
 */
public class RagManager {

    private static final Logger LOG = LoggerFactory.getLogger(RagManager.class);

    public static final int DEFAULT_TOP_K = 5;

    /**
     * Context window used when embedding chunks/queries. Deliberately smaller than the
     * typical chat-model default (4096) — chunks are capped at ~400 words (~500-600
     * tokens) and KV-cache allocation cost scales with nCtx, not input length, so a
     * small fixed value keeps per-chunk embedding overhead low.
     */
    private static final int EMBED_N_CTX     = 1024;
    private static final int EMBED_N_THREADS = Math.max(1, Runtime.getRuntime().availableProcessors());

    private final Path ragDir;
    private final ModelRegistry registry;

    /** Loaded embedding models, keyed by registry model name. Loaded once, kept for process lifetime. */
    private final Map<String, LlamaModel> embedModels = new ConcurrentHashMap<>();

    /** Collections for which a query-time native-library-unavailable warning has already been logged. */
    private final Map<String, Boolean> degradedWarned = new ConcurrentHashMap<>();

    public RagManager(Path ragDir, ModelRegistry registry) {
        this.ragDir = ragDir;
        this.registry = registry;
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    /**
     * Index a file or directory into the named collection (BM25-only).
     * Re-indexing an already-indexed file replaces its previous chunks.
     * Unsupported or unreadable files are skipped with a warning.
     *
     * @param collection collection name (created automatically if it doesn't exist)
     * @param path       file or directory to index
     * @param progress   optional callback that receives one status line per file
     */
    public void addDocuments(String collection, Path path, Consumer<String> progress)
            throws Exception {
        addDocuments(collection, path, null, progress);
    }

    /**
     * Index a file or directory into the named collection, optionally computing and
     * storing vector embeddings alongside the BM25 index.
     *
     * <p>If {@code embedModel} is non-null and the collection has no prior
     * {@link HybridConfig} (i.e. this is the first time hybrid mode is enabled), any
     * existing documents are dropped and the collection is rebuilt from scratch — this
     * avoids ever leaving a collection with partial vector coverage. If the collection
     * is already hybrid, {@code embedModel} may be omitted (the recorded model is reused)
     * or must match the recorded model exactly.
     *
     * @param collection collection name (created automatically if it doesn't exist)
     * @param path       file or directory to index
     * @param embedModel registry name of the embedding model to use, or {@code null} for BM25-only
     * @param progress   optional callback that receives one status line per file
     */
    public void addDocuments(String collection, Path path, String embedModel, Consumer<String> progress)
            throws Exception {
        Files.createDirectories(ragDir);
        Path indexPath = ragDir.resolve(collection);

        HybridConfig existing = HybridConfig.load(indexPath);
        LlamaModel embedder = null;
        HybridConfig effectiveConfig = existing;

        if (embedModel != null && existing != null && !embedModel.equals(existing.embedModel)) {
            throw new IllegalArgumentException(
                "Collection '" + collection + "' was built with embedding model '" + existing.embedModel
                + "', not '" + embedModel + "'. Run 'jllm rag rm " + collection
                + "' and re-add to switch embedding models.");
        }

        String resolvedEmbedModel = embedModel != null ? embedModel : (existing != null ? existing.embedModel : null);

        if (resolvedEmbedModel != null) {
            if (!LlamaModel.isNativeLibraryAvailable()) {
                throw new IllegalStateException(
                    "Hybrid RAG indexing requires the native JNI library, which is not available.");
            }
            embedder = loadEmbedModel(resolvedEmbedModel);

            boolean firstTimeHybrid = existing == null;
            if (firstTimeHybrid && Files.exists(indexPath) && new RagIndex(indexPath).docCount() > 0) {
                // Enabling hybrid on a collection that already has BM25-only documents:
                // rebuild fully rather than risk partial vector coverage.
                deleteCollection(collection);
                Files.createDirectories(indexPath);
            }
            if (firstTimeHybrid) {
                float[] probe = embedder.embed("probe", EMBED_N_CTX, EMBED_N_THREADS);
                effectiveConfig = new HybridConfig(resolvedEmbedModel, probe.length);
            }
        }

        List<Path> files;
        if (Files.isDirectory(path)) {
            try (Stream<Path> stream = Files.walk(path)) {
                files = stream.filter(Files::isRegularFile)
                              .filter(DocumentReader::isSupported)
                              .sorted()
                              .collect(Collectors.toList());
            }
            if (files.isEmpty()) {
                throw new IllegalArgumentException(
                    "No supported files found under: " + path);
            }
        } else {
            if (!DocumentReader.isSupported(path)) {
                throw new IllegalArgumentException(
                    "Unsupported file type: " + path.getFileName()
                    + "  (supported: .pdf, .txt, .md, .java, .py, .json, ...)");
            }
            files = List.of(path);
        }

        // Open one IndexWriter for the entire batch — much faster than open/close per file.
        LlamaModel embedderForBatch = embedder;
        try (IndexWriter writer = RagIndex.openWriter(indexPath)) {
            for (Path file : files) {
                indexFile(writer, file, embedderForBatch, progress);
            }
            writer.commit();
        }

        if (resolvedEmbedModel != null && effectiveConfig != null) {
            effectiveConfig.save(indexPath);
        }
    }

    private void indexFile(IndexWriter writer, Path path, LlamaModel embedder, Consumer<String> progress) {
        String absPath = path.toAbsolutePath().toString();
        try {
            // Delete previous chunks for this source so re-indexing is idempotent.
            writer.deleteDocuments(new Term("source", absPath));

            List<DocumentReader.PageContent> pages = DocumentReader.readPages(path);
            int totalChunks = 0;
            for (DocumentReader.PageContent page : pages) {
                List<String> chunks = DocumentChunker.chunk(page.text);
                for (String chunk : chunks) {
                    float[] embedding = null;
                    if (embedder != null) {
                        try {
                            embedding = embedder.embed(chunk, EMBED_N_CTX, EMBED_N_THREADS);
                        } catch (Exception e) {
                            LOG.warn("Failed to embed a chunk of {}: {}", path, e.getMessage());
                        }
                    }
                    writer.addDocument(RagIndex.buildDocument(absPath, page.page, chunk, embedding));
                    totalChunks++;
                }
            }
            if (progress != null) {
                progress.accept(String.format("  Indexed: %-40s  %d chunk(s)",
                        path.getFileName(), totalChunks));
            }
        } catch (Exception e) {
            LOG.warn("Failed to index {}: {}", path, e.getMessage());
            if (progress != null) {
                progress.accept(String.format("  Skipped: %-40s  (%s)",
                        path.getFileName(), e.getMessage()));
            }
        }
    }

    private LlamaModel loadEmbedModel(String modelName) {
        ModelConfig cfg = registry.get(modelName)
                .orElseThrow(() -> new IllegalArgumentException("Unknown embed model: " + modelName));
        return embedModels.computeIfAbsent(modelName, n -> {
            int nGpuLayers = cfg.getNumGpuLayers() != null ? cfg.getNumGpuLayers() : 0;
            return new LlamaModel(cfg.getPath(), nGpuLayers);
        });
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /**
     * Retrieve the top {@value #DEFAULT_TOP_K} most relevant chunks for {@code query}
     * from the named collection.  Returns an empty list if the collection doesn't exist.
     */
    public List<RagResult> search(String collection, String query) throws Exception {
        return search(collection, query, DEFAULT_TOP_K);
    }

    /**
     * Retrieve the top {@code topK} most relevant chunks for {@code query}. Transparently
     * performs hybrid (BM25+vector) search if the collection was indexed with an
     * embedding model, otherwise performs plain BM25 search — callers do not need to know
     * which mode a collection is in.
     */
    public List<RagResult> search(String collection, String query, int topK) throws Exception {
        Path indexPath = ragDir.resolve(collection);
        RagIndex index = new RagIndex(indexPath);

        HybridConfig hybrid = HybridConfig.load(indexPath);
        if (hybrid == null) {
            return index.search(query, topK);
        }

        if (!LlamaModel.isNativeLibraryAvailable()) {
            if (degradedWarned.putIfAbsent(collection, Boolean.TRUE) == null) {
                LOG.warn("Collection '{}' is hybrid but the native JNI library is unavailable; "
                        + "falling back to BM25-only search.", collection);
            }
            return index.search(query, topK);
        }

        try {
            LlamaModel embedder = loadEmbedModel(hybrid.embedModel);
            float[] queryEmbedding = embedder.embed(query, EMBED_N_CTX, EMBED_N_THREADS);
            return index.search(query, topK, queryEmbedding);
        } catch (Exception e) {
            LOG.warn("Vector search failed for collection '{}', falling back to BM25-only: {}",
                    collection, e.getMessage());
            return index.search(query, topK);
        }
    }

    // ── Collection management ─────────────────────────────────────────────────

    /** List all collections with their chunk counts. */
    public List<CollectionInfo> listCollections() throws Exception {
        if (!Files.exists(ragDir)) return Collections.emptyList();
        List<CollectionInfo> result = new ArrayList<>();
        try (Stream<Path> stream = Files.list(ragDir)) {
            for (Path dir : stream.filter(Files::isDirectory)
                                  .collect(Collectors.toList())) {
                try {
                    int count = new RagIndex(dir).docCount();
                    HybridConfig hybrid = HybridConfig.load(dir);
                    String embedModel = hybrid != null ? hybrid.embedModel : null;
                    result.add(new CollectionInfo(dir.getFileName().toString(), dir, count, embedModel));
                } catch (Exception e) {
                    LOG.debug("Skipping invalid index dir: {}", dir);
                }
            }
        }
        result.sort(Comparator.comparing(c -> c.name));
        return result;
    }

    /** Delete a collection and all its index files. */
    public void deleteCollection(String collection) throws Exception {
        Path indexPath = ragDir.resolve(collection);
        if (!Files.exists(indexPath)) {
            throw new IllegalArgumentException("Collection not found: " + collection);
        }
        try (Stream<Path> stream = Files.walk(indexPath)) {
            stream.sorted(Comparator.reverseOrder())
                  .forEach(p -> {
                      try { Files.delete(p); }
                      catch (Exception e) { LOG.warn("Failed to delete {}", p); }
                  });
        }
    }

    // ── RAG prompt building ───────────────────────────────────────────────────

    /**
     * Build a context block from retrieved chunks to prepend to the system prompt.
     * Returns {@code null} if {@code hits} is empty.
     */
    public static String buildContextBlock(List<RagResult> hits) {
        if (hits == null || hits.isEmpty()) return null;
        StringBuilder sb = new StringBuilder();
        sb.append("[Context from local documents — use this to answer the user's question.");
        sb.append(" If it lacks relevant info, say so.]\n");
        for (RagResult r : hits) {
            sb.append("---\n");
            // Show only the filename, not the full path, to keep the prompt readable.
            String fileName = Path.of(r.source).getFileName().toString();
            sb.append("Source: ").append(fileName);
            if (r.page > 0) sb.append(" (page ").append(r.page).append(")");
            sb.append("\n").append(r.content).append("\n");
        }
        sb.append("---");
        return sb.toString();
    }

    // ── CollectionInfo ────────────────────────────────────────────────────────

    public static class CollectionInfo {
        public final String name;
        public final Path   path;
        public final int    chunkCount;

        /** Registry name of the embedding model, or {@code null} for BM25-only collections. */
        public final String embedModel;

        CollectionInfo(String name, Path path, int chunkCount, String embedModel) {
            this.name       = name;
            this.path       = path;
            this.chunkCount = chunkCount;
            this.embedModel = embedModel;
        }
    }
}
