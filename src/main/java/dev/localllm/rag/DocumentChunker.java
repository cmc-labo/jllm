package dev.localllm.rag;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Splits document text into overlapping word-level chunks for indexing.
 *
 * <p>Chunks are formed by sliding a fixed-size window over the word array.
 * The window advances by {@code (chunkWords - overlapWords)} words each step,
 * so adjacent chunks share {@code overlapWords} words of context. When possible,
 * the chunk boundary is pushed back to a sentence-ending word ({@code .}, {@code ?},
 * {@code !}) to avoid cutting mid-sentence.
 */
public class DocumentChunker {

    static final int CHUNK_WORDS   = 400;
    static final int OVERLAP_WORDS = 50;

    public static List<String> chunk(String text) {
        return chunk(text, CHUNK_WORDS, OVERLAP_WORDS);
    }

    public static List<String> chunk(String text, int chunkWords, int overlapWords) {
        String normalized = text.replaceAll("\\s+", " ").trim();
        if (normalized.isEmpty()) return new ArrayList<>();

        String[] words = normalized.split(" ");
        List<String> chunks = new ArrayList<>();
        int start = 0;

        while (start < words.length) {
            int end = Math.min(start + chunkWords, words.length);

            // Try to snap end to a sentence boundary within the last ~30 words.
            if (end < words.length) {
                int snap = findSentenceBoundary(words,
                        Math.max(start + overlapWords, end - 30), end);
                if (snap > start) end = snap;
            }

            String chunk = String.join(" ", Arrays.copyOfRange(words, start, end)).trim();
            if (!chunk.isEmpty()) chunks.add(chunk);

            if (end >= words.length) break;
            start = Math.max(start + 1, end - overlapWords);
        }
        return chunks;
    }

    private static int findSentenceBoundary(String[] words, int from, int to) {
        for (int i = to - 1; i >= from; i--) {
            String w = words[i];
            if (w.endsWith(".") || w.endsWith("!") || w.endsWith("?")
             || w.endsWith("。") || w.endsWith("！") || w.endsWith("？")) {
                return i + 1;
            }
        }
        return to;
    }
}
