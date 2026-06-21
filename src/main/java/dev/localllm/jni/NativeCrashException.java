package dev.localllm.jni;

/**
 * Thrown when native code (llama.cpp / ggml) hit a fatal signal (segfault,
 * abort from a failed {@code GGML_ASSERT}, bus error, etc.) that the JNI
 * layer intercepted and converted into a Java exception instead of letting
 * it kill the JVM process.
 *
 * <p>Catching this is not the same as recovering cleanly: a SIGSEGV/SIGBUS
 * in particular may leave native heap state corrupted in ways that are not
 * visible from Java. Once this exception has been thrown once, every
 * subsequent native call in this process will immediately throw it again
 * without re-entering native code - the process should be restarted rather
 * than continuing to serve requests indefinitely.
 */
public final class NativeCrashException extends RuntimeException {

    public NativeCrashException(String message) {
        super(message);
    }
}
