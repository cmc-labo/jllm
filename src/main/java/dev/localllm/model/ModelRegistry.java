package dev.localllm.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class ModelRegistry {

    private static final Path REGISTRY_DIR  = Paths.get(System.getProperty("user.home"), ".local-llm");
    private static final Path REGISTRY_FILE = REGISTRY_DIR.resolve("models.json");
    private static final Path MANAGED_DIR   = REGISTRY_DIR.resolve("models");
    private static final Path PLUGINS_DIR   = REGISTRY_DIR.resolve("plugins");

    /** Returns the directory where managed GGUF files are stored. */
    public static Path getManagedModelsDir() { return MANAGED_DIR; }

    /** Returns the directory where plugin JARs are stored. */
    public static Path getPluginsDir() { return PLUGINS_DIR; }
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final Type MODEL_LIST_TYPE = new TypeToken<List<ModelConfig>>() {}.getType();

    public List<ModelConfig> list() {
        if (!Files.exists(REGISTRY_FILE)) {
            return new ArrayList<>();
        }
        try {
            String json = Files.readString(REGISTRY_FILE);
            List<ModelConfig> models = GSON.fromJson(json, MODEL_LIST_TYPE);
            return models != null ? models : new ArrayList<>();
        } catch (IOException e) {
            System.err.println("Warning: Could not read registry: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    public void add(ModelConfig model) throws IOException {
        List<ModelConfig> models = list();
        models.removeIf(m -> m.getName().equals(model.getName()));
        models.add(model);
        save(models);
    }

    public boolean remove(String name) {
        List<ModelConfig> models = list();
        boolean removed = models.removeIf(m -> m.getName().equals(name));
        if (removed) {
            try {
                save(models);
            } catch (IOException e) {
                System.err.println("Warning: Could not save registry: " + e.getMessage());
            }
        }
        return removed;
    }

    public Optional<ModelConfig> get(String name) {
        return list().stream().filter(m -> m.getName().equals(name)).findFirst();
    }

    private void save(List<ModelConfig> models) throws IOException {
        Files.createDirectories(REGISTRY_DIR);
        Files.writeString(REGISTRY_FILE, GSON.toJson(models));
    }
}
