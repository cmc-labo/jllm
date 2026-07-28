package dev.localllm.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ModelfileTest {

    @Test
    void appliesFromAndParameterInstructions() {
        ModelConfig config = new ModelConfig();
        Modelfile.apply(
                "FROM /path/to/model.gguf\n" +
                "PARAMETER temperature 0.7\n" +
                "PARAMETER num_predict 512\n" +
                "PARAMETER num_ctx 4096\n",
                config);

        assertEquals("/path/to/model.gguf", config.getPath());
        assertEquals(0.7f, config.getTemperature());
        assertEquals(512, config.getNumPredict());
        assertEquals(4096, config.getNumCtx());
    }

    @Test
    void ignoresBlankLinesAndComments() {
        ModelConfig config = new ModelConfig();
        Modelfile.apply(
                "# a comment\n" +
                "\n" +
                "FROM /path/to/model.gguf\n" +
                "# another comment\n",
                config);

        assertEquals("/path/to/model.gguf", config.getPath());
    }

    @Test
    void ignoresUnknownInstructionsForForwardCompatibility() {
        ModelConfig config = new ModelConfig();
        Modelfile.apply(
                "FROM /path/to/model.gguf\n" +
                "TEMPLATE {{ .Prompt }}\n" +
                "ADAPTER /path/to/adapter\n",
                config);

        // Should not throw, and the recognised instruction still takes effect.
        assertEquals("/path/to/model.gguf", config.getPath());
    }

    @Test
    void invalidParameterValueIsIgnoredNotThrown() {
        ModelConfig config = new ModelConfig();
        Modelfile.apply("PARAMETER temperature not-a-number\n", config);
        assertNull(config.getTemperature());
    }

    @Test
    void parsesMultiLineSystemBlock() {
        ModelConfig config = new ModelConfig();
        Modelfile.apply(
                "SYSTEM \"\"\"\n" +
                "You are a helpful assistant.\n" +
                "Always answer in Japanese.\n" +
                "\"\"\"\n",
                config);

        assertEquals("You are a helpful assistant.\nAlways answer in Japanese.",
                config.getSystemPrompt());
    }

    @Test
    void toTextRoundTripsThroughApply() {
        ModelConfig original = new ModelConfig();
        original.setPath("/path/to/model.gguf");
        original.setTemperature(0.8f);
        original.setNumPredict(256);
        original.setSystemPrompt("Be concise.");

        String text = Modelfile.toText(original);

        ModelConfig reparsed = new ModelConfig();
        Modelfile.apply(text, reparsed);

        assertEquals(original.getPath(), reparsed.getPath());
        assertEquals(original.getTemperature(), reparsed.getTemperature());
        assertEquals(original.getNumPredict(), reparsed.getNumPredict());
        assertEquals(original.getSystemPrompt(), reparsed.getSystemPrompt());
    }
}
