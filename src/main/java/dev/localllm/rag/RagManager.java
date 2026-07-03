package dev.localllm.rag;

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
 * <p>Usage flow:
 * <ol>
 *   <li>Index documents: {@code jllm rag add <collection> <path>}</li>
 *   <li>Chat with RAG: {@code jllm run <model> --rag <collection>}</li>
 *   <li>Or via API: include {@code "rag_collection": "<name>"} in the request body.</li>
 * </ol>
 */
public class RagManager {

    private static final Logger LOG = LoggerFactory.getLogger(RagManager.class);

    public static final int DEFAULT_TOP_K = 5;

    private final Path ragDir;

    public RagManager(Path ragDir) {
        this.ragDir = ragDir;
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    /**
     * Index a file or directory into the named collection.
     * Re-indexing an already-indexed file replaces its previous chunks.
     * Unsupported or unreadable files are skipped with a warning.
     *
     * @param collection collection name (created automatically if it doesn't exist)
     * @param path       file or directory to index
     * @param progress   optional callback that receives one status line per file
     */
    public void addDocuments(String collection, Path path, Consumer<String> progress)
            throws Exception {
        Files.createDirectories(ragDir);
        Path indexPath = ragDir.resolve(collection);

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
        try (IndexWriter writer = RagIndex.openWriter(indexPath)) {
            for (Path file : files) {
                indexFile(writer, file, progress);
            }
            writer.commit();
        }
    }

    private void indexFile(IndexWriter writer, Path path, Consumer<String> progress) {
        String absPath = path.toAbsolutePath().toString();
        try {
            // Delete previous chunks for this source so re-indexing is idempotent.
            writer.deleteDocuments(new Term("source", absPath));

            List<DocumentReader.PageContent> pages = DocumentReader.readPages(path);
            int totalChunks = 0;
            for (DocumentReader.PageContent page : pages) {
                List<String> chunks = DocumentChunker.chunk(page.text);
                for (String chunk : chunks) {
                    writer.addDocument(RagIndex.buildDocument(absPath, page.page, chunk));
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

    // ── Search ────────────────────────────────────────────────────────────────

    /**
     * Retrieve the top {@value #DEFAULT_TOP_K} most relevant chunks for {@code query}
     * from the named collection.  Returns an empty list if the collection doesn't exist.
     */
    public List<RagResult> search(String collection, String query) throws Exception {
        return search(collection, query, DEFAULT_TOP_K);
    }

    public List<RagResult> search(String collection, String query, int topK) throws Exception {
        Path indexPath = ragDir.resolve(collection);
        return new RagIndex(indexPath).search(query, topK);
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
                    result.add(new CollectionInfo(dir.getFileName().toString(), dir, count));
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

        CollectionInfo(String name, Path path, int chunkCount) {
            this.name       = name;
            this.path       = path;
            this.chunkCount = chunkCount;
        }
    }
}
