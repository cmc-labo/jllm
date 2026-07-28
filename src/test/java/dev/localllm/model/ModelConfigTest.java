package dev.localllm.model;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelConfigTest {

    @Test
    void isSplitIsFalseWhenShardsIsNull() {
        ModelConfig config = new ModelConfig();
        config.setShards(null);
        assertFalse(config.isSplit());
    }

    @Test
    void isSplitIsFalseWithASingleShard() {
        ModelConfig config = new ModelConfig();
        config.setShards(Collections.singletonList("model-00001-of-00001.gguf"));
        assertFalse(config.isSplit());
    }

    @Test
    void isSplitIsTrueWithMultipleShards() {
        ModelConfig config = new ModelConfig();
        config.setShards(Arrays.asList(
                "model-00001-of-00002.gguf",
                "model-00002-of-00002.gguf"));
        assertTrue(config.isSplit());
    }

    @Test
    void gettersReturnWhatWasSet() {
        ModelConfig config = new ModelConfig();
        config.setName("mymodel");
        config.setPath("/models/mymodel.gguf");
        config.setTemperature(0.5f);
        config.setNumCtx(8192);

        assertEquals("mymodel", config.getName());
        assertEquals("/models/mymodel.gguf", config.getPath());
        assertEquals(0.5f, config.getTemperature());
        assertEquals(8192, config.getNumCtx());
    }
}
