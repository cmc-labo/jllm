package dev.localllm.rag;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DocumentChunkerTest {

    @Test
    void emptyTextProducesNoChunks() {
        assertEquals(0, DocumentChunker.chunk("").size());
        assertEquals(0, DocumentChunker.chunk("   \n  ").size());
    }

    @Test
    void shortTextFitsInASingleChunk() {
        List<String> chunks = DocumentChunker.chunk("one two three four five.");
        assertEquals(1, chunks.size());
        assertEquals("one two three four five.", chunks.get(0));
    }

    @Test
    void slidesAFixedWindowWithOverlapOverLongerText() {
        StringBuilder text = new StringBuilder();
        for (int i = 1; i <= 25; i++) text.append("w").append(i).append(' ');

        List<String> chunks = DocumentChunker.chunk(text.toString().trim(), 10, 2);

        assertEquals(3, chunks.size());
        assertEquals("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10", chunks.get(0));
        assertEquals("w9 w10 w11 w12 w13 w14 w15 w16 w17 w18", chunks.get(1));
        assertEquals("w17 w18 w19 w20 w21 w22 w23 w24 w25", chunks.get(2));
    }

    @Test
    void collapsesInternalWhitespace() {
        List<String> chunks = DocumentChunker.chunk("one\n\ttwo   three");
        assertEquals(1, chunks.size());
        assertEquals("one two three", chunks.get(0));
    }

    @Test
    void everyChunkIsNonEmpty() {
        StringBuilder text = new StringBuilder();
        for (int i = 1; i <= 900; i++) text.append("word").append(i).append(' ');

        List<String> chunks = DocumentChunker.chunk(text.toString().trim());
        assertTrue(chunks.size() > 1, "expected more than one chunk for 900 words");
        for (String c : chunks) {
            assertTrue(!c.isEmpty(), "chunk must not be empty");
        }
    }
}
