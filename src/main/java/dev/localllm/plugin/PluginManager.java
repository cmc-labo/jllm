package dev.localllm.plugin;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Discovers and loads plugin JARs from a directory (default: {@code ~/.local-llm/plugins/}).
 *
 * <p>Each JAR is loaded in its own {@link URLClassLoader} whose parent is the
 * application classloader. This isolates plugins from each other (separate
 * class namespaces) while still allowing them to use the
 * {@code dev.localllm.plugin.*} interfaces from the host application.
 *
 * <p>JARs must expose their implementations via the Java SPI mechanism
 * ({@code META-INF/services/} descriptor files). JARs are loaded in
 * alphabetical order; {@link PromptInterceptor}s are sorted by
 * {@link PromptInterceptor#getPriority()} after all JARs are scanned.
 *
 * <p>{@link #load()} is idempotent: calling it again clears and reloads all
 * plugins (useful for hot-reload during development).
 */
public class PluginManager {

    /** An empty manager with no tools or interceptors. Returned by the no-arg runner constructor. */
    public static final PluginManager EMPTY = new PluginManager(null) {
        @Override public void load() { /* nothing */ }
    };

    private final Path pluginDir;
    private final List<LlmTool> tools             = new ArrayList<>();
    private final List<PromptInterceptor> interceptors = new ArrayList<>();

    // maps a plugin instance back to its source JAR filename (for display)
    private final Map<Object, String> sourceJar = new LinkedHashMap<>();

    public PluginManager(Path pluginDir) {
        this.pluginDir = pluginDir;
    }

    /**
     * Scan the plugin directory and load all JARs. Clears previously loaded
     * plugins before reloading. Safe to call more than once.
     */
    public void load() {
        tools.clear();
        interceptors.clear();
        sourceJar.clear();

        if (pluginDir == null || !Files.isDirectory(pluginDir)) return;

        File[] jars = pluginDir.toFile().listFiles(f -> f.getName().endsWith(".jar"));
        if (jars == null || jars.length == 0) return;

        Arrays.sort(jars, Comparator.comparing(File::getName));
        for (File jar : jars) loadJar(jar);

        interceptors.sort(Comparator.comparingInt(PromptInterceptor::getPriority));
    }

    private void loadJar(File jar) {
        try {
            URLClassLoader cl = new URLClassLoader(
                    new URL[]{jar.toURI().toURL()},
                    Thread.currentThread().getContextClassLoader());

            for (LlmTool tool : ServiceLoader.load(LlmTool.class, cl)) {
                tools.add(tool);
                sourceJar.put(tool, jar.getName());
            }
            for (PromptInterceptor ic : ServiceLoader.load(PromptInterceptor.class, cl)) {
                interceptors.add(ic);
                sourceJar.put(ic, jar.getName());
            }
        } catch (Exception e) {
            System.err.println("Warning: failed to load plugin " + jar.getName() + ": " + e.getMessage());
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /** All loaded tools in load order. Immutable view. */
    public List<LlmTool> getTools() {
        return Collections.unmodifiableList(tools);
    }

    /** All loaded interceptors sorted by priority. Immutable view. */
    public List<PromptInterceptor> getInterceptors() {
        return Collections.unmodifiableList(interceptors);
    }

    /** The source JAR filename for a given plugin instance, or {@code "unknown"}. */
    public String getSourceJar(Object plugin) {
        return sourceJar.getOrDefault(plugin, "unknown");
    }

    /** The directory that was scanned, or {@code null} for {@link #EMPTY}. */
    public Path getPluginDir() {
        return pluginDir;
    }

    public boolean hasTools()        { return !tools.isEmpty(); }
    public boolean hasInterceptors() { return !interceptors.isEmpty(); }

    /**
     * Apply the full interceptor chain to a prompt and return the result.
     * Returns the prompt unchanged if no interceptors are loaded.
     */
    public String applyInterceptors(String prompt) {
        for (PromptInterceptor ic : interceptors) {
            prompt = ic.intercept(prompt);
        }
        return prompt;
    }
}
