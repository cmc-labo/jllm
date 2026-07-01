import dev.localllm.plugin.LlmTool;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
 * Example jllm plugin: echoes back the text the model passes in the args.
 *
 * Build:
 *   javac -cp /path/to/local-llm.jar EchoTool.java
 *   jar cf echo-tool.jar EchoTool.class META-INF/
 *
 * Install:
 *   mkdir -p ~/.local-llm/plugins/
 *   cp echo-tool.jar ~/.local-llm/plugins/
 *
 * Then run:
 *   jllm plugins          # should show "echo" in the tool list
 *   jllm run <model>      # ask "repeat the word hello" to trigger it
 */
public class EchoTool implements LlmTool {

    @Override
    public String getName() {
        return "echo";
    }

    @Override
    public String getDescription() {
        return "Echoes the provided text back verbatim.";
    }

    @Override
    public String getParametersSchema() {
        return "{\"type\":\"object\","
             + "\"properties\":{\"text\":{\"type\":\"string\",\"description\":\"The text to echo\"}},"
             + "\"required\":[\"text\"]}";
    }

    @Override
    public String execute(String argsJson) throws Exception {
        JsonObject args = new Gson().fromJson(argsJson, JsonObject.class);
        if (!args.has("text")) throw new IllegalArgumentException("Missing required parameter: text");
        return args.get("text").getAsString();
    }
}
