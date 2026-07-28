package dev.localllm.rag;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RankFusionTest {

    @Test
    void ranksResultsHigherWhenBothSignalsAgree() {
        RagResult a = new RagResult("a.txt", -1, "A content", 0f);
        RagResult b = new RagResult("b.txt", -1, "B content", 0f);
        RagResult c = new RagResult("c.txt", -1, "C content", 0f);

        // b is ranked 2nd by BM25 but 1st by vector search, so RRF should place it first.
        List<RagResult> bm25   = Arrays.asList(a, b, c);
        List<RagResult> vector = Arrays.asList(b, c, a);

        List<RagResult> fused = RankFusion.fuse(bm25, vector, 10);

        assertEquals(3, fused.size());
        assertEquals("b.txt", fused.get(0).source);
        assertEquals("a.txt", fused.get(1).source);
        assertEquals("c.txt", fused.get(2).source);
    }

    @Test
    void resultAppearingInBothListsCanOutrankABetterSingleListRank() {
        RagResult topBm25Only   = new RagResult("d.txt", -1, "D content", 0f);
        RagResult inBothAtRank5 = new RagResult("e.txt", -1, "E content", 0f);

        List<RagResult> bm25 = Arrays.asList(
                topBm25Only,
                new RagResult("x1", -1, "x1", 0f),
                new RagResult("x2", -1, "x2", 0f),
                new RagResult("x3", -1, "x3", 0f),
                inBothAtRank5);
        List<RagResult> vector = Arrays.asList(
                new RagResult("y1", -1, "y1", 0f),
                new RagResult("y2", -1, "y2", 0f),
                new RagResult("y3", -1, "y3", 0f),
                new RagResult("y4", -1, "y4", 0f),
                inBothAtRank5);

        List<RagResult> fused = RankFusion.fuse(bm25, vector, 10);

        assertEquals("e.txt", fused.get(0).source,
                "a result present in both ranked lists should beat one present in only one, "
                + "even at a worse individual rank");
    }

    @Test
    void truncatesToTopK() {
        RagResult a = new RagResult("a.txt", -1, "A content", 0f);
        RagResult b = new RagResult("b.txt", -1, "B content", 0f);
        RagResult c = new RagResult("c.txt", -1, "C content", 0f);

        List<RagResult> fused = RankFusion.fuse(Arrays.asList(a, b, c), Arrays.asList(b, c, a), 2);
        assertEquals(2, fused.size());
    }

    @Test
    void aResultFoundInOnlyOneListHasNaNForTheOtherSignal() {
        RagResult bm25Only = new RagResult("bm25-only.txt", -1, "content", 0f);

        List<RagResult> fused = RankFusion.fuse(
                Arrays.asList(bm25Only), Arrays.asList(), 10);

        assertEquals(1, fused.size());
        assertTrue(Float.isNaN(fused.get(0).vectorScore));
    }
}
