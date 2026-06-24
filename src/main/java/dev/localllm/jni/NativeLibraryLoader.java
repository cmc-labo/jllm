package dev.localllm.jni;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Locates and loads {@code libllamajni.so} (or the platform-equivalent
 * {@code .dylib} / {@code .dll}).
 *
 * <p>Resolution order:
 * <ol>
 *   <li>{@code -Ddev.localllm.nativeLib=<absolute-path>} &mdash; exact file</li>
 *   <li>{@code -Ddev.localllm.nativeLibDir=<dir>} &mdash; directory; uses the
 *       platform-specific filename within it</li>
 *   <li>{@code ./native/build/<filename>} relative to the working directory
 *       (dev mode &mdash; present after running {@code native/build.sh})</li>
 *   <li>Bundled inside the JAR at {@code native/<classifier>/<filename>}, where
 *       {@code <classifier>} encodes the target OS, CPU architecture, and
 *       optional GPU variant (e.g. {@code linux-x86_64-cuda}). GPU variants are
 *       tried before the CPU-only fallback; see {@link #probeOrder}.</li>
 * </ol>
 *
 * <h3>Classifier naming scheme</h3>
 * <pre>
 *   {os}-{arch}[-{gpu}]
 *
 *   os    : linux | osx | windows
 *   arch  : x86_64 | aarch64
 *   gpu   : cuda | rocm | metal   (omitted for CPU-only builds)
 * </pre>
 * Build a variant with {@code native/build.sh --static --variant cuda} (etc.)
 * to produce a self-contained shared library that can be bundled into the JAR.
 * See {@code native/build.sh --help} for full usage.
 */
final class NativeLibraryLoader {

    private static final Logger LOG = LoggerFactory.getLogger(NativeLibraryLoader.class);

    // Version used to namespace the temp-dir extraction path, so parallel
    // installs of different versions don't collide.  Must match the version
    // stamped into the fat JAR's manifest when/if version tracking is added.
    private static final String VERSION = "1.0.0";

    private static final AtomicBoolean LOADED = new AtomicBoolean(false);

    private NativeLibraryLoader() {}

    static void load() {
        if (!LOADED.compareAndSet(false, true)) {
            return;
        }

        String os = detectOs();
        String filename = libFilename(os);

        // 1. Explicit file path override.
        String explicitLib = System.getProperty("dev.localllm.nativeLib");
        if (explicitLib != null) {
            Path p = Paths.get(explicitLib).toAbsolutePath().normalize();
            LOG.debug("Loading native library from system property: {}", p);
            System.load(p.toString());
            return;
        }

        // 2. Explicit directory override; platform filename within it.
        String explicitDir = System.getProperty("dev.localllm.nativeLibDir");
        if (explicitDir != null) {
            Path p = Paths.get(explicitDir, filename).toAbsolutePath().normalize();
            LOG.debug("Loading native library from nativeLibDir: {}", p);
            System.load(p.toString());
            return;
        }

        // 3. Dev-mode path: ./native/build/<filename> relative to CWD.
        Path devPath = Paths.get("native", "build", filename);
        if (Files.exists(devPath)) {
            try {
                Path abs = devPath.toAbsolutePath().normalize();
                LOG.debug("Loading native library from dev path: {}", abs);
                System.load(abs.toString());
                return;
            } catch (UnsatisfiedLinkError e) {
                LOG.debug("Dev-path load failed ({}); falling through to JAR extraction", e.getMessage());
            }
        }

        // 4. Extract from JAR and load.
        loadFromJar(os, detectArch(), filename);
    }

    private static void loadFromJar(String os, String arch, String filename) {
        List<String> toTry = probeOrder(os, arch);
        List<String> tried = new ArrayList<>();

        for (String classifier : toTry) {
            String resource = "native/" + classifier + "/" + filename;
            if (NativeLibraryLoader.class.getClassLoader().getResource(resource) == null) {
                LOG.debug("Variant not bundled in JAR, skipping: {}", classifier);
                continue;
            }
            Path extracted;
            try {
                extracted = extractToTemp(resource, filename, classifier);
            } catch (IOException e) {
                LOG.warn("Failed to extract {}: {}", resource, e.getMessage());
                tried.add(classifier + " (extraction error: " + e.getMessage() + ")");
                continue;
            }
            try {
                LOG.info("Loading native library ({}) from JAR", classifier);
                System.load(extracted.toString());
                return;
            } catch (UnsatisfiedLinkError e) {
                LOG.warn("Load failed for {}: {}", extracted, e.getMessage());
                tried.add(classifier + " (load error: " + e.getMessage() + ")");
            }
        }

        throw new UnsatisfiedLinkError(
            "Cannot load " + filename + " for " + os + "-" + arch + ". " +
            (tried.isEmpty()
                ? "No matching variant found in JAR. Bundle native/" + os + "-" + arch + "/"
                  + filename + " inside the JAR (see native/build.sh --static)."
                : "Variants tried: " + tried + ".") + " " +
            "Use -Ddev.localllm.nativeLib=<path> to point to an external library."
        );
    }

    /**
     * Returns the ordered list of JAR classifiers to probe for this platform.
     * GPU-accelerated variants come first; the base CPU-only classifier is
     * always the final fallback. Only classifiers whose resources exist in the
     * JAR will actually be loaded (the caller skips missing ones).
     *
     * <p>Package-private for testing.
     */
    static List<String> probeOrder(String os, String arch) {
        List<String> candidates = new ArrayList<>();
        String base = os + "-" + arch;

        switch (os) {
            case "linux":
                if (hasCuda())  candidates.add(base + "-cuda");
                if (hasRocm())  candidates.add(base + "-rocm");
                break;
            case "osx":
                // Apple Silicon: Metal is bundled into osx-aarch64 builds by default,
                // so no separate device detection is needed.  An explicit -metal
                // variant is tried first in case a build with more Metal-specific
                // tuning is available.
                if ("aarch64".equals(arch)) {
                    candidates.add(base + "-metal");
                }
                break;
            case "windows":
                if (hasCuda()) candidates.add(base + "-cuda");
                break;
        }

        candidates.add(base); // CPU-only (or implicit-Metal on macOS) fallback
        return candidates;
    }

    private static Path extractToTemp(String resource, String filename, String classifier)
            throws IOException {
        Path dir = Paths.get(
            System.getProperty("java.io.tmpdir"), "local-llm-native", VERSION, classifier);
        Files.createDirectories(dir);
        Path dest = dir.resolve(filename);

        try (InputStream in = NativeLibraryLoader.class.getClassLoader()
                                                        .getResourceAsStream(resource)) {
            if (in == null) {
                throw new IOException("Resource disappeared from JAR: " + resource);
            }
            // REPLACE_EXISTING: safe because same VERSION dir always holds the
            // same binary; a previous partial write is the only risk we avoid.
            Files.copy(in, dest, StandardCopyOption.REPLACE_EXISTING);
        }
        return dest;
    }

    // ── platform detection ────────────────────────────────────────────────

    static String detectOs() {
        String name = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (name.contains("win"))                      return "windows";
        if (name.contains("mac") || name.contains("darwin")) return "osx";
        return "linux";
    }

    static String detectArch() {
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        if ("aarch64".equals(arch) || "arm64".equals(arch)) return "aarch64";
        return "x86_64";
    }

    static String libFilename(String os) {
        switch (os) {
            case "windows": return "llamajni.dll";
            case "osx":     return "libllamajni.dylib";
            default:        return "libllamajni.so";
        }
    }

    // ── GPU detection ─────────────────────────────────────────────────────

    private static boolean hasCuda() {
        switch (detectOs()) {
            case "linux":
                // /proc/driver/nvidia/version is created by the nvidia kernel module.
                return new java.io.File("/proc/driver/nvidia/version").exists();
            case "windows":
                // nvcuda.dll is installed by the NVIDIA display driver.
                String sysRoot = System.getenv("SystemRoot");
                if (sysRoot == null) sysRoot = "C:\\Windows";
                return new java.io.File(sysRoot, "System32\\nvcuda.dll").exists();
            default:
                return false;
        }
    }

    private static boolean hasRocm() {
        // /dev/kfd is created by the ROCm kernel fusion driver (amdkfd module).
        return new java.io.File("/dev/kfd").exists();
    }
}
