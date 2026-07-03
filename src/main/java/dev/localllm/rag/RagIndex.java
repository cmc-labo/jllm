package dev.localllm.rag;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Thin wrapper around a single Lucene {@link FSDirectory} index.
 *
 * <p>Write operations (indexing) are handled by {@link RagManager} which opens
 * an {@link IndexWriter} once per batch.  This class exposes read-only search
 * and statistics operations, each of which opens and closes the directory and
 * reader within a single method call so they are safe to call from multiple
 * threads simultaneously.
 *
 * <p>Index schema per document (chunk):
 * <ul>
 *   <li>{@code source} — absolute path of the source file (StringField, stored, indexed)</li>
 *   <li>{@code page}   — 1-based page number; -1 for plain-text files (StoredField)</li>
 *   <li>{@code content} — chunk text (TextField, analyzed + stored, BM25-ranked)</li>
 * </ul>
 */
public class RagIndex {

    private final Path indexPath;

    public RagIndex(Path indexPath) {
        this.indexPath = indexPath;
    }

    /**
     * Search for the {@code topK} most relevant chunks using BM25 (Lucene default).
     * Returns an empty list if the index does not exist or the query fails.
     */
    public List<RagResult> search(String queryText, int topK) throws Exception {
        if (!indexExists()) return Collections.emptyList();

        try (FSDirectory dir = FSDirectory.open(indexPath);
             DirectoryReader reader = DirectoryReader.open(dir)) {

            IndexSearcher searcher = new IndexSearcher(reader);
            StandardAnalyzer analyzer = new StandardAnalyzer();
            QueryParser parser = new QueryParser("content", analyzer);
            parser.setDefaultOperator(QueryParser.Operator.OR);

            Query query;
            try {
                query = parser.parse(QueryParser.escape(queryText));
            } catch (Exception e) {
                return Collections.emptyList();
            }

            TopDocs hits = searcher.search(query, topK);
            List<RagResult> results = new ArrayList<>();
            StoredFields sf = searcher.storedFields();

            for (ScoreDoc sd : hits.scoreDocs) {
                Document doc = sf.document(sd.doc);
                String source  = doc.get("source");
                String content = doc.get("content");
                int page = -1;
                org.apache.lucene.index.IndexableField pageField = doc.getField("page");
                if (pageField != null && pageField.numericValue() != null) {
                    page = pageField.numericValue().intValue();
                }
                results.add(new RagResult(source, page, content, sd.score));
            }
            return results;
        }
    }

    /** Total number of indexed chunks (documents) in this collection. */
    public int docCount() throws Exception {
        if (!indexExists()) return 0;
        try (FSDirectory dir = FSDirectory.open(indexPath);
             DirectoryReader reader = DirectoryReader.open(dir)) {
            return reader.numDocs();
        }
    }

    private boolean indexExists() {
        if (!Files.exists(indexPath)) return false;
        try (FSDirectory dir = FSDirectory.open(indexPath)) {
            return DirectoryReader.indexExists(dir);
        } catch (Exception e) {
            return false;
        }
    }

    // ── Package-private write helpers (used by RagManager) ───────────────────

    /** Open an IndexWriter for this collection (CREATE_OR_APPEND). Caller must close. */
    static IndexWriter openWriter(Path indexPath) throws Exception {
        Files.createDirectories(indexPath);
        StandardAnalyzer analyzer = new StandardAnalyzer();
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
        return new IndexWriter(FSDirectory.open(indexPath), config);
    }

    /** Build a Lucene Document for one text chunk. */
    static Document buildDocument(String absSource, int page, String chunkText) {
        Document doc = new Document();
        doc.add(new StringField("source", absSource, Field.Store.YES));
        doc.add(new StoredField("page", page));
        doc.add(new TextField("content", chunkText, Field.Store.YES));
        return doc;
    }
}
