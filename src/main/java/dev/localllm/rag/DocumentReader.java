package dev.localllm.rag;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Reads documents (PDF and plain text) into page-level text units for chunking.
 *
 * <p>Supported formats:
 * <ul>
 *   <li><b>PDF</b> ({@code .pdf}) — text is extracted per page via Apache PDFBox.
 *       Scanned/image-only PDFs produce no text and are silently skipped.</li>
 *   <li><b>Plain text</b> — any extension listed in {@link #TEXT_EXTENSIONS}, or any
 *       filename listed in {@link #TEXT_FILENAMES} (for common extensionless files
 *       like {@code Dockerfile}), is read as UTF-8 and returned as a single page.</li>
 * </ul>
 */
public class DocumentReader {

    static final Set<String> TEXT_EXTENSIONS = Set.of(
        // Docs / prose
        ".txt", ".md", ".markdown", ".rst", ".adoc", ".tex",
        // Programming languages
        ".java", ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rb", ".php", ".swift", ".kt", ".kts",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".scala", ".rs", ".lua", ".pl", ".r", ".dart",
        ".ex", ".exs", ".clj", ".hs", ".elm", ".vue", ".svelte", ".m", ".mm", ".groovy",
        // Web / styling
        ".html", ".htm", ".xml", ".css", ".scss", ".less",
        // Data / config
        ".json", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
        ".csv", ".tsv", ".log", ".properties",
        // Build / infra / schema
        ".sh", ".bat", ".ps1", ".sql", ".gradle", ".proto", ".graphql", ".gql", ".tf", ".tfvars"
    );

    /** Filenames (compared case-insensitively, no extension) recognized as plain text. */
    static final Set<String> TEXT_FILENAMES = Set.of(
        "dockerfile", "makefile", "jenkinsfile", "rakefile", "gemfile",
        ".gitignore", ".dockerignore"
    );

    public static class PageContent {
        /** 1-based page number (always 1 for non-PDF files). */
        public final int page;
        public final String text;

        PageContent(int page, String text) {
            this.page = page;
            this.text = text;
        }
    }

    /**
     * Returns true if the file has a supported extension and is a regular file.
     * Used as a filter predicate during directory walks.
     */
    public static boolean isSupported(Path path) {
        if (!Files.isRegularFile(path)) return false;
        String name = path.getFileName().toString().toLowerCase();
        if (name.endsWith(".pdf")) return true;
        if (TEXT_FILENAMES.contains(name)) return true;
        for (String ext : TEXT_EXTENSIONS) {
            if (name.endsWith(ext)) return true;
        }
        return false;
    }

    /**
     * Reads a document and returns its text split into pages.
     * PDFs produce one entry per non-empty page; text files produce a single entry.
     */
    public static List<PageContent> readPages(Path path) throws Exception {
        String name = path.getFileName().toString().toLowerCase();
        if (name.endsWith(".pdf")) return readPdf(path);
        return readText(path);
    }

    private static List<PageContent> readPdf(Path path) throws Exception {
        List<PageContent> pages = new ArrayList<>();
        try (PDDocument doc = Loader.loadPDF(path.toFile())) {
            PDFTextStripper stripper = new PDFTextStripper();
            int numPages = doc.getNumberOfPages();
            for (int i = 1; i <= numPages; i++) {
                stripper.setStartPage(i);
                stripper.setEndPage(i);
                String text = stripper.getText(doc).trim();
                if (!text.isEmpty()) {
                    pages.add(new PageContent(i, text));
                }
            }
        }
        return pages;
    }

    private static List<PageContent> readText(Path path) throws Exception {
        String text = Files.readString(path);
        List<PageContent> result = new ArrayList<>();
        if (!text.isBlank()) result.add(new PageContent(1, text));
        return result;
    }
}
