package dev.localllm.jni;

/**
 * Minimal smoke test for the JNI binding.
 * Usage: java -Ddev.localllm.nativeLibDir=native/build -cp target/classes \
 *   dev.localllm.jni.LlamaDemo <model.gguf> [prompt]
 */
public final class LlamaDemo {
    private LlamaDemo() {}

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: LlamaDemo <model.gguf> [prompt]");
            System.exit(1);
        }
        String modelPath = args[0];
        String prompt = args.length > 1 ? args[1] : "Once upon a time";

        try (LlamaModel model = new LlamaModel(modelPath, 0);
             LlamaContext ctx = model.createContext(256, 2)) {

            System.out.print(prompt);
            System.out.flush();
            ctx.generateStreaming(prompt, 64, 0.0f, piece -> {
                System.out.print(piece);
                System.out.flush();
            });
            System.out.println();
        }
    }
}
