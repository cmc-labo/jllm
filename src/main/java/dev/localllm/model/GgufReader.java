package dev.localllm.model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Reads the KV metadata section from a GGUF binary file.
 *
 * <p>Only the header and KV pairs are read; tensor data is never accessed.
 * Supports GGUF format versions 1, 2, and 3.
 *
 * <p>Key fields extracted:
 * <ul>
 *   <li>{@code general.architecture} → {@link GgufMetadata#architecture}</li>
 *   <li>{@code general.file_type}    → {@link GgufMetadata#quantization}</li>
 *   <li>{@code general.parameter_count} → {@link GgufMetadata#parameterCount}</li>
 *   <li>{@code {arch}.context_length} → {@link GgufMetadata#contextLength}</li>
 *   <li>{@code {arch}.block_count}   → {@link GgufMetadata#blockCount}</li>
 *   <li>{@code {arch}.embedding_length} → {@link GgufMetadata#embeddingLength}</li>
 * </ul>
 */
public class GgufReader {

    // "GGUF" as a little-endian uint32: bytes G(0x47) G(0x47) U(0x55) F(0x46)
    private static final int GGUF_MAGIC = 0x46554747;

    // GGUF value types
    private static final int T_UINT8  = 0,  T_INT8   = 1,  T_UINT16 = 2,  T_INT16  = 3;
    private static final int T_UINT32 = 4,  T_INT32  = 5,  T_FLOAT  = 6,  T_BOOL   = 7;
    private static final int T_STRING = 8,  T_ARRAY  = 9,  T_UINT64 = 10, T_INT64  = 11;
    private static final int T_DOUBLE = 12;

    // general.file_type enum → quantization name
    private static final String[] QUANT_NAMES = {
        "F32",       //  0
        "F16",       //  1
        "Q4_0",      //  2
        "Q4_1",      //  3
        null,        //  4  (deprecated)
        null,        //  5  (deprecated)
        null,        //  6  (deprecated)
        "Q8_0",      //  7
        "Q5_0",      //  8
        "Q5_1",      //  9
        "Q2_K",      // 10
        "Q3_K_S",    // 11
        "Q3_K_M",    // 12
        "Q3_K_L",    // 13
        "Q4_K_S",    // 14
        "Q4_K_M",    // 15
        "Q5_K_S",    // 16
        "Q5_K_M",    // 17
        "Q6_K",      // 18
        "Q8_K",      // 19
        "IQ2_XXS",   // 20
        "IQ2_XS",    // 21
        "IQ3_XXS",   // 22
        "IQ1_S",     // 23
        "IQ4_NL",    // 24
        "IQ3_S",     // 25
        "IQ3_M",     // 26
        "IQ2_S",     // 27
        "IQ2_M",     // 28
        "IQ4_XS",    // 29
        "IQ1_M",     // 30
        "BF16",      // 31
        "Q4_0_4_4",  // 32
        "Q4_0_4_8",  // 33
        "Q4_0_8_8",  // 34
    };

    // ── result type ───────────────────────────────────────────────────────────

    public static final class GgufMetadata {
        /** Value of {@code general.architecture} (e.g. "llama", "phi3", "mistral"). */
        public String  architecture;
        /** Quantization name derived from {@code general.file_type} (e.g. "Q4_K_M", "BF16"). */
        public String  quantization;
        /** Raw value of {@code general.parameter_count}. */
        public Long    parameterCount;
        /** Model's native context window from {@code {arch}.context_length}. */
        public Integer contextLength;
        /** Number of transformer layers from {@code {arch}.block_count}. */
        public Integer blockCount;
        /** Hidden dimension from {@code {arch}.embedding_length}. */
        public Integer embeddingLength;

        /** Format {@link #parameterCount} as a human-readable label, e.g. {@code "7.24B"}. */
        public String paramCountLabel() {
            if (parameterCount == null || parameterCount <= 0) return null;
            double p = parameterCount;
            if (p >= 1e12) return String.format("%.2fT", p / 1e12);
            if (p >= 1e9)  return String.format("%.2fB", p / 1e9);
            if (p >= 1e6)  return String.format("%.1fM", p / 1e6);
            return Long.toString(parameterCount);
        }
    }

    // ── instance state ─────────────────────────────────────────────────────────

    private final FileChannel ch;
    // 64 KB read-ahead buffer; filled lazily
    private final ByteBuffer  buf = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN);

    private GgufReader(FileChannel ch) {
        this.ch = ch;
        buf.limit(0); // start empty
    }

    // ── public API ──────────────────────────────────────────────────────────────

    /**
     * Read the GGUF KV metadata section from {@code path}.
     * Returns a partially-populated {@link GgufMetadata}; fields that are absent in the
     * file remain {@code null}.
     *
     * @throws IOException if the file cannot be read or is not a valid GGUF file
     */
    public static GgufMetadata read(Path path) throws IOException {
        try (FileChannel c = FileChannel.open(path, StandardOpenOption.READ)) {
            return new GgufReader(c).parse();
        }
    }

    // ── parsing ─────────────────────────────────────────────────────────────────

    private GgufMetadata parse() throws IOException {
        if (readInt() != GGUF_MAGIC) {
            throw new IOException("Not a GGUF file (invalid magic)");
        }
        long version = readUInt32();
        if (version < 1 || version > 3) {
            throw new IOException("Unsupported GGUF version: " + version);
        }

        // v1 uses uint32 for header counts; v2+ uses uint64
        @SuppressWarnings("unused")
        long nTensors = (version == 1) ? readUInt32() : readLong();
        long nKv      = (version == 1) ? readUInt32() : readLong();

        GgufMetadata meta = new GgufMetadata();
        for (long i = 0; i < nKv; i++) {
            String key     = readString(256);       // keys are always short
            int    valType = (int) readUInt32();
            consumeKv(key, valType, meta);
        }
        return meta;
    }

    /**
     * For keys we care about, read the value into {@code meta}.
     * For everything else, skip the value without touching meta.
     */
    private void consumeKv(String key, int type, GgufMetadata meta) throws IOException {
        if ("general.architecture".equals(key) && type == T_STRING) {
            meta.architecture = readString(256);
            return;
        }
        if ("general.file_type".equals(key) && isU32Like(type)) {
            meta.quantization = quantName(readFourAsLong());
            return;
        }
        if ("general.parameter_count".equals(key) && isU64Like(type)) {
            meta.parameterCount = readLong();
            return;
        }
        // Architecture-prefixed uint32 keys (work without knowing the arch prefix)
        if (isU32Like(type)) {
            if (key.endsWith(".context_length"))   { meta.contextLength   = (int) readFourAsLong(); return; }
            if (key.endsWith(".block_count"))      { meta.blockCount      = (int) readFourAsLong(); return; }
            if (key.endsWith(".embedding_length")) { meta.embeddingLength = (int) readFourAsLong(); return; }
        }
        skipValue(type);
    }

    private void skipValue(int type) throws IOException {
        switch (type) {
            case T_UINT8:  case T_INT8:  case T_BOOL:    skip(1); break;
            case T_UINT16: case T_INT16:                  skip(2); break;
            case T_UINT32: case T_INT32: case T_FLOAT:   skip(4); break;
            case T_UINT64: case T_INT64: case T_DOUBLE:  skip(8); break;
            case T_STRING:
                skip(readLong());  // uint64 length, then that many bytes
                break;
            case T_ARRAY: {
                int  elemType = (int) readUInt32();
                long count    = readLong();
                int  fixed    = fixedSize(elemType);
                if (fixed > 0) {
                    skip((long) fixed * count);
                } else {
                    for (long i = 0; i < count; i++) skipValue(elemType);
                }
                break;
            }
            default:
                throw new IOException("Unknown GGUF value type: " + type);
        }
    }

    // ── primitive I/O ──────────────────────────────────────────────────────────

    /**
     * Ensure at least {@code n} bytes are buffered, refilling from the channel as needed.
     */
    private void ensure(int n) throws IOException {
        while (buf.remaining() < n) {
            buf.compact();
            int read = ch.read(buf);
            buf.flip();
            if (read < 0 && buf.remaining() < n) {
                throw new IOException("Unexpected end of GGUF file");
            }
        }
    }

    private int readInt() throws IOException {
        ensure(4);
        return buf.getInt();
    }

    private long readUInt32() throws IOException {
        return Integer.toUnsignedLong(readInt());
    }

    /** Read 4 bytes as an unsigned 32-bit value (works for both T_UINT32 and T_INT32). */
    private long readFourAsLong() throws IOException {
        return readUInt32();
    }

    private long readLong() throws IOException {
        ensure(8);
        return buf.getLong();
    }

    private String readString(int maxLen) throws IOException {
        long len = readLong();
        if (len < 0 || len > maxLen) {
            throw new IOException("GGUF string length out of range: " + len);
        }
        int    ilen = (int) len;
        byte[] b    = new byte[ilen];
        int    off  = 0;
        while (off < ilen) {
            if (!buf.hasRemaining()) ensure(1);
            int n = Math.min(buf.remaining(), ilen - off);
            buf.get(b, off, n);
            off += n;
        }
        return new String(b, StandardCharsets.UTF_8);
    }

    /**
     * Skip {@code n} bytes.
     *
     * <p>Bytes already in the buffer are consumed in place; remaining bytes are
     * skipped by seeking the underlying {@link FileChannel} directly, so even
     * large skips (e.g. over a vocabulary array) are O(1).
     */
    private void skip(long n) throws IOException {
        long fromBuf = Math.min(n, buf.remaining());
        buf.position(buf.position() + (int) fromBuf);
        long rest = n - fromBuf;
        if (rest > 0) {
            // ch.position() is just past the end of what was read into buf.
            // Our current logical position = ch.position() - buf.remaining().
            // New logical position = current + rest.
            ch.position(ch.position() - buf.remaining() + rest);
            buf.limit(0); // invalidate stale buffer content
        }
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    private static boolean isU32Like(int t) { return t == T_UINT32 || t == T_INT32; }
    private static boolean isU64Like(int t) { return t == T_UINT64 || t == T_INT64; }

    /** Returns the byte size of fixed-width types, or 0 for variable-width (STRING, ARRAY). */
    private static int fixedSize(int type) {
        switch (type) {
            case T_UINT8:  case T_INT8:  case T_BOOL:   return 1;
            case T_UINT16: case T_INT16:                 return 2;
            case T_UINT32: case T_INT32: case T_FLOAT:  return 4;
            case T_UINT64: case T_INT64: case T_DOUBLE: return 8;
            default: return 0;
        }
    }

    private static String quantName(long ft) {
        if (ft >= 0 && ft < QUANT_NAMES.length && QUANT_NAMES[(int) ft] != null) {
            return QUANT_NAMES[(int) ft];
        }
        return "Q" + ft;
    }
}
