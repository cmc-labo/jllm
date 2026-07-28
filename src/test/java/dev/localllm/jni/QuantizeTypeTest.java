package dev.localllm.jni;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QuantizeTypeTest {

    @Test
    void fromStringIsCaseInsensitive() {
        assertEquals(QuantizeType.Q4_K_M, QuantizeType.fromString("q4_k_m"));
        assertEquals(QuantizeType.Q4_K_M, QuantizeType.fromString("Q4_K_M"));
    }

    @Test
    void fromStringNormalizesDashesToUnderscores() {
        assertEquals(QuantizeType.Q4_K_M, QuantizeType.fromString("Q4-K-M"));
    }

    @Test
    void fromStringReturnsNullForUnknownOrNullInput() {
        assertNull(QuantizeType.fromString("not-a-real-type"));
        assertNull(QuantizeType.fromString(null));
    }

    @Test
    void validNamesListsEveryEnumConstant() {
        String names = QuantizeType.validNames();
        for (QuantizeType t : QuantizeType.values()) {
            assertTrue(names.contains(t.name), "validNames() should mention " + t.name);
        }
    }
}
