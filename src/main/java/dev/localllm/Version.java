package dev.localllm;

/**
 * Compile-time version constants for jllm and its bundled dependencies.
 * Keep these in sync with the versions declared in build.sh.
 */
public final class Version {

    public static final String JLLM     = "0.1.0";

    // ── Bundled dependency versions ───────────────────────────────────────────
    public static final String GSON     = "2.10.1";
    public static final String SLF4J    = "2.0.13";
    public static final String LOGBACK  = "1.5.6";
    public static final String UNDERTOW = "2.3.14.Final";
    public static final String XNIO     = "3.8.14.Final";
    public static final String LUCENE   = "9.11.1";
    public static final String PDFBOX   = "3.0.3";

    private Version() {}
}
