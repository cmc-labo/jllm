package dev.localllm.jni;

import java.lang.ref.Cleaner;

/** Single shared {@link Cleaner} for all native handles in this package. */
final class NativeCleaner {
    static final Cleaner INSTANCE = Cleaner.create();

    private NativeCleaner() {}
}
