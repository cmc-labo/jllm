package dev.localllm.jni;

import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Forwards llama.cpp/ggml native log output (model loading, hardware /
 * backend detection, decode warnings, ...) into SLF4J instead of letting
 * it print directly to stdout/stderr, so it can be filtered, formatted,
 * and routed like any other application log via the configured SLF4J
 * binding (e.g. Logback).
 */
final class NativeLogBridge {

    private static final Logger LOG = LoggerFactory.getLogger("dev.localllm.native");
    private static final AtomicBoolean INSTALLED = new AtomicBoolean(false);

    private NativeLogBridge() {}

    /** Registers the bridge once per process. Call before {@link LlamaNative#backendInit()}. */
    static void installOnce() {
        if (!INSTALLED.compareAndSet(false, true)) {
            return;
        }
        LlamaNative.setLogCallback(NativeLogBridge::onLog);
    }

    private static void onLog(int level, String message) {
        switch (level) {
            case LlamaNative.LOG_LEVEL_DEBUG:
                LOG.debug(message);
                break;
            case LlamaNative.LOG_LEVEL_WARN:
                LOG.warn(message);
                break;
            case LlamaNative.LOG_LEVEL_ERROR:
                LOG.error(message);
                break;
            case LlamaNative.LOG_LEVEL_INFO:
            default:
                LOG.info(message);
                break;
        }
    }
}
