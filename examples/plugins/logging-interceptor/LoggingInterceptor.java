import dev.localllm.plugin.PromptInterceptor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;

/**
 * Example jllm prompt interceptor: appends every assembled prompt to a log file.
 *
 * Build:
 *   javac -cp /path/to/local-llm.jar LoggingInterceptor.java
 *   jar cf logging-interceptor.jar LoggingInterceptor.class META-INF/
 *
 * Install:
 *   mkdir -p ~/.local-llm/plugins/
 *   cp logging-interceptor.jar ~/.local-llm/plugins/
 *
 * Prompts are appended to: ~/.local-llm/prompt.log
 */
public class LoggingInterceptor implements PromptInterceptor {

    private static final Path LOG_FILE =
            Paths.get(System.getProperty("user.home"), ".local-llm", "prompt.log");

    @Override
    public int getPriority() {
        return 100; // run last so the log captures what the model actually receives
    }

    @Override
    public String intercept(String prompt) {
        try {
            String entry = "=== " + Instant.now() + " ===\n" + prompt + "\n\n";
            Files.writeString(LOG_FILE, entry,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.err.println("[LoggingInterceptor] Failed to write log: " + e.getMessage());
        }
        return prompt; // pass through unchanged
    }
}
