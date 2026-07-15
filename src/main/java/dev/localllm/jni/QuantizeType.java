package dev.localllm.jni;

/**
 * Common llama_ftype quantization targets supported by {@link LlamaNative#quantize}.
 * Values match the {@code LLAMA_FTYPE_MOSTLY_*} constants in llama.h.
 */
public enum QuantizeType {
    Q2_K   ("Q2_K",   10),
    Q3_K_S ("Q3_K_S", 11),
    Q3_K_M ("Q3_K_M", 12),
    Q3_K_L ("Q3_K_L", 13),
    Q4_0   ("Q4_0",    2),
    Q4_K_S ("Q4_K_S", 14),
    Q4_K_M ("Q4_K_M", 15),
    Q5_0   ("Q5_0",    8),
    Q5_K_S ("Q5_K_S", 16),
    Q5_K_M ("Q5_K_M", 17),
    Q6_K   ("Q6_K",   18),
    Q8_0   ("Q8_0",    7),
    F16    ("F16",     1),
    BF16   ("BF16",   32);

    public final String name;
    public final int ftypeId;

    QuantizeType(String name, int ftypeId) {
        this.name    = name;
        this.ftypeId = ftypeId;
    }

    /** Case-insensitive lookup; returns {@code null} if not found. */
    public static QuantizeType fromString(String s) {
        if (s == null) return null;
        String upper = s.toUpperCase(java.util.Locale.ROOT).replace('-', '_');
        for (QuantizeType t : values()) {
            if (t.name.equals(upper)) return t;
        }
        return null;
    }

    /** Human-readable list of supported names for error messages. */
    public static String validNames() {
        StringBuilder sb = new StringBuilder();
        for (QuantizeType t : values()) {
            if (sb.length() > 0) sb.append(", ");
            sb.append(t.name);
        }
        return sb.toString();
    }
}
