package dev.localllm.runner;

import dev.localllm.model.ModelConfig;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class ModelRunner {

    public void runInteractive(ModelConfig model) throws Exception {
        if (model.getBinary() == null) {
            throw new RuntimeException(
                "No binary configured for '" + model.getName() + "'. " +
                "Re-register with: local-llm add " + model.getName() +
                " --path <path> --binary <llama-cli>"
            );
        }

        List<String> cmd = new ArrayList<>();
        cmd.add(model.getBinary());
        cmd.add("-m"); cmd.add(model.getPath());
        cmd.add("-i");
        cmd.add("--chat-template"); cmd.add("chatml");
        cmd.add("-c"); cmd.add(model.getNumCtx() != null ? String.valueOf(model.getNumCtx()) : "4096");
        if (model.getNumThreads() != null) {
            cmd.add("-t"); cmd.add(String.valueOf(model.getNumThreads()));
        }
        if (model.getTemperature() != null) {
            cmd.add("--temp"); cmd.add(String.valueOf(model.getTemperature()));
        }
        if (model.getSystemPrompt() != null && !model.getSystemPrompt().isEmpty()) {
            cmd.add("--system"); cmd.add(model.getSystemPrompt());
        }

        System.out.println("Starting " + model.getName() + "  (Ctrl+C to quit)");
        System.out.println();

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        int exitCode = pb.start().waitFor();
        if (exitCode != 0) {
            System.err.println("Model exited with code: " + exitCode);
        }
    }

    // Used by the HTTP API server for single-shot generation.
    public String generate(ModelConfig model, String prompt, int maxTokens) throws Exception {
        if (model.getBinary() == null) {
            throw new RuntimeException("No binary configured for '" + model.getName() + "'.");
        }

        List<String> cmd = new ArrayList<>();
        cmd.add(model.getBinary());
        cmd.add("-m"); cmd.add(model.getPath());
        cmd.add("-p"); cmd.add(prompt);
        cmd.add("-n"); cmd.add(String.valueOf(maxTokens));
        cmd.add("--no-display-prompt");

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(false);
        Process process = pb.start();

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("llama process exited with code: " + exitCode);
        }
        return output.toString().trim();
    }
}
