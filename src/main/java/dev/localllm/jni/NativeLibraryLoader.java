package dev.localllm.jni;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Locates and loads {@code libllamajni.so}. The JNI library is linked with an
 * rpath pointing at the llama.cpp shared libraries it depends on
 * (libllama.so, libggml*.so), so loading it by absolute path is sufficient -
 * java.library.path does not need to be configured.
 *
 * Search order:
 *   1. -Ddev.localllm.nativeLib=/path/to/libllamajni.so (exact file)
 *   2. -Ddev.localllm.nativeLibDir=/path/to/dir (directory containing it)
 *   3. ./native/build/libllamajni.so (relative to the working directory, dev mode)
 */
final class NativeLibraryLoader {

    private static final String LIB_FILE_NAME = "libllamajni.so";
    private static final AtomicBoolean LOADED = new AtomicBoolean(false);

    private NativeLibraryLoader() {}

    static void load() {
        if (!LOADED.compareAndSet(false, true)) {
            return;
        }

        String explicitLib = System.getProperty("dev.localllm.nativeLib");
        if (explicitLib != null) {
            System.load(Paths.get(explicitLib).toAbsolutePath().normalize().toString());
            return;
        }

        String explicitDir = System.getProperty("dev.localllm.nativeLibDir");
        if (explicitDir != null) {
            System.load(Paths.get(explicitDir, LIB_FILE_NAME).toAbsolutePath().normalize().toString());
            return;
        }

        Path devPath = Paths.get("native", "build", LIB_FILE_NAME);
        if (Files.exists(devPath)) {
            try {
                System.load(devPath.toAbsolutePath().normalize().toString());
                return;
            } catch (UnsatisfiedLinkError e) {
                throw new UnsatisfiedLinkError(
                    "Found " + devPath + " but failed to load it: " + e.getMessage());
            }
        }

        throw new UnsatisfiedLinkError(
            "Could not find " + LIB_FILE_NAME + ". Build it with native/build.sh, or set " +
            "-Ddev.localllm.nativeLib=<path-to-so> / -Ddev.localllm.nativeLibDir=<dir>."
        );
    }
}
